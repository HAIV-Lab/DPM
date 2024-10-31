import os.path as osp
from collections import OrderedDict
import torch
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import build_transform
import torch.nn as nn
from copy import deepcopy
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import sklearn.metrics as sk
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
from utils.utils import CosineClassifier

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "ImageNet": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
}


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_kl(id_sim_mean, logits):
    _score = []
    # print(logits)
    # print(id_sim_mean)
    scores = pairwise_distances(logits, id_sim_mean, metric=kl)  # Batch * 100
    # print(scores.shape)
    train_images_targets = np.eye(id_sim_mean.shape[0])  # id_sim_lables
    kl_logits = (-scores) @ train_images_targets

    return kl_logits


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda").train()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,  # 256
            nhead,  # 4
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):  # men [B,L,256] x[B, 49, 256]
        q = k = v = self.norm1(x)  # [B, L, 256]
        x = x + self.self_attn(q, k, v)  # [B, L, 256]
        q = self.norm2(x)  # [B, L, 256]
        x = x + self.cross_attn(q, mem, mem)  # [B, L, 256]
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x  # [B, 49, 256]


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class DPM_Block(nn.Module):
    def __init__(self, text_features,
                 input_dim, num_classes):  # input_dim=512
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pre_project_s = Proj2()  # (B, D)
        self.pre_project_t = Proj1()
        self.pre_project_vv = Proj1()
        self.scale = input_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * 30., requires_grad=False)
        self.CosClassifier = CosineClassifier(num_classes, input_dim)
        self.vis_gamma_p = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.vis_gamma_n = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.visual_prototype = nn.Parameter(text_features.clone().detach())  # , requires_grad=False)

    def forward(self, Fs, Ft, Fv, label):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  # [B, 2L, 49]
        A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
        feat_v_a = A_weight1 @ Fs  # [B, L, C]
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  # [B, L, C] + Fv
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  # [B, 2L, 49]
        A_weight1v = F.softmax(A_weightv, dim=-1)  # [B, 2L, 49]
        feat_v_av = A_weight1v @ Fs  # [B, L, C]
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  # [B, L, C] + Fv
        Ft = F.normalize(Ft, dim=-1, p=2)  # [L, C]
        Fv = F.normalize(Fv, dim=-1, p=2)  # [B, L, C]
        feat_v = F.normalize(feat_v, dim=-1, p=2)  # [B, L, C]
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  # [B, L, C]
        logits1 = torch.mul(Fv, Ft).sum(-1)
        logits2 = torch.mul(feat_v, Ft).sum(-1)
        logits3 = torch.mul(feat_vv, self.visual_prototype).sum(-1)
        with torch.no_grad():
            class_count = torch.bincount(label, minlength=L)
            class_sum = Fv[:, 0, :].new_zeros(L, C)
            class_sum.index_add_(0, label, Fv[:, 0, :])
            safe_class_count = class_count.float().unsqueeze(1).clamp_min(1e-8)
            class_mean = class_sum / safe_class_count
            mask = class_count > 0
            new_visual_prototype = 0.99 * self.visual_prototype + 0.01 * class_mean
            updated_visual_prototype = self.visual_prototype.clone()
            updated_visual_prototype[mask] = new_visual_prototype[mask]
            self.visual_prototype.data = updated_visual_prototype
        return logits1, logits2, logits3

    def evaluate(self, Fs, Ft, Fv):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  # [B, 2L, 49]
        A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
        feat_v_a = A_weight1 @ Fs  # [B, L, C]
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  # [B, L, C] + Fv
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  # [B, 2L, 49]
        A_weight1v = F.softmax(A_weightv, dim=-1)  # [B, 2L, 49]
        feat_v_av = A_weight1v @ Fs  # [B, L, C]
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  # [B, L, C] + Fv
        Ft = F.normalize(Ft, dim=-1, p=2)  # [L, C]
        Fv = F.normalize(Fv, dim=-1, p=2)  # [B, L, C]
        feat_v = F.normalize(feat_v, dim=-1, p=2)  # [B, L, C]
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  # [B, L, C]
        logits1 = self.logit_scale * torch.mul(Fv, Ft).sum(-1)
        logits2 = self.logit_scale * torch.mul(feat_v, Ft).sum(-1)
        logits3 = self.logit_scale * torch.mul(feat_vv, self.visual_prototype).sum(-1)
        return logits1, logits2, logits3


class DPM_T(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        with torch.no_grad():
            temp = CUSTOM_TEMPLATES['cifar100']
            prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.dpmt = DPM_Block(text_features, 512, len(classnames))

    def forward(self, image, label, cls_id=None):
        image_features, local_features = self.image_encoder(
            image.type(self.dtype))  # image_features, [B, C], local [B, 49, C]
        prompts, tokenized_prompts = self.prompt_learner(cls_id)  # prompts [2*L, 77, D]  tokenized_prompts  [2*L, 77]
        text_features = self.text_encoder(prompts, tokenized_prompts)  # text_features [2*L, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [2*L, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # image_f [B, C]
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  # local [B, 49, C]
        logits1, logits2, logits3 = self.dpmt(Fs=local_features, Ft=text_features, Fv=image_features,
                                              label=label)  # .squeeze()
        return logits1, logits2, logits3

    def evaluate(self, image, cls_id=None):
        image_features, local_features = self.image_encoder(
            image.type(self.dtype))  # image_features, [B, C], local [B, 49, C]
        prompts, tokenized_prompts = self.prompt_learner(
            cls_id)  # prompts [2*L, 77, D]  tokenized_prompts  [2*L, 77]
        text_features = self.text_encoder(prompts, tokenized_prompts)  # text_features [2*L, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [2*L, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # image_f [B, C]
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  # local [B, 49, C]
        logits1, logits2, logits3 = self.dpmt.evaluate(Fs=local_features, Ft=text_features,
                                                       Fv=image_features)  # .squeeze()
        return logits1, logits2, logits3


@TRAINER_REGISTRY.register()
class DPM(TrainerX):  # need to be trained

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPM.PREC in ["fp16", "fp32"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DPM.PREC == "fp32":
            clip_model.float()
        print("Building custom CLIP")
        self.model = DPM_T(cfg, classnames, clip_model)  # (cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "dpmt" not in name:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.dpmt, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("dpmt", self.model.dpmt, self.optim, self.sched)
        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)
        self.scaler = None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model.image_encoder = nn.DataParallel(self.model.image_encoder)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        self.celoss = nn.CrossEntropyLoss()
        self.celoss.to(self.device)
        self.img_match = torch.zeros(len(classnames), len(classnames)).to(self.device)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        custom_tfm_train += [tfm_train]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        logits1, logits2, logits3 = self.model(image, label)  
        # loss_1 = self.celoss(20 * logits1, label)  # [B, L]
        loss_2 = self.celoss(20 * logits2, label)  # [B, L]
        loss_3 = self.celoss(20 * logits3, label)
        loss = loss_2 + loss_3
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            # 'loss_1': loss_1.item(),
            'loss_2': loss_2.item(),
            'loss_3': loss_3.item(),
            "acc": compute_accuracy(logits2, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input1 = batch["img"]
        label = batch["label"]
        label = label.to(self.device)
        return input1.to(self.device), label

    def train(self, train_loader, id_loader, ood_loader_list):  # , start_epoch, max_epoch):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            auroc = 1.0
            if self.epoch > 4:
                auroc = self.test_ood(train_loader, id_loader, ood_loader_list)
            # self.after_epoch(auroc)
            self.save_model(self.epoch, self.output_dir)
        self.test_ood(train_loader, id_loader, ood_loader_list)

    def after_epoch(self, auroc):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
        curr_result = auroc  # self.test(split="val")
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=curr_result,
                model_name="model-best.pth.tar"
            )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def test_ood(self, train_loader, id_loader, ood_loader_list, method='maxlogits'):
        self.class_sim_mean = None
        in_fea1, in_fea2, in_fea3, in_kl = self.get_feature(train_loader, id_loader)

        label_gt = id_loader.dataset.targets
        label_pre = np.argmax(in_fea1, axis=1)
        metric = [label_pre[i] == label_gt[i] for i in range(len(label_pre))]
        print('acc_logit1', np.sum(metric) / len(metric))
        label_pre = np.argmax(in_fea2, axis=1)
        metric = [label_pre[i] == label_gt[i] for i in range(len(label_pre))]
        print('acc_logit2', np.sum(metric) / len(metric))
        label_pre = np.argmax(in_fea3, axis=1)
        metric = [label_pre[i] == label_gt[i] for i in range(len(label_pre))]
        print('acc_logit3', np.sum(metric) / len(metric))

        ood_name = ['iNaturalist', 'SUN', 'places365', 'dtd']  # ['SUN', 'places365'] #
        mean_auroc = 0.0
        for i in range(len(ood_name)):
            print('*******',ood_name[i],'*******')
            ood_loader = ood_loader_list[i]
            out_fea1, out_fea2, out_fea3, out_kl = self.get_feature(train_loader, ood_loader)

            def _scale(x, target_min, target_max):
                y = (x - x.min()) / (x.max() - x.min())
                y *= target_max - target_min
                y += target_min
                return y

            target_max, target_min = max(in_fea2.max(), out_fea2.max()), min(in_fea2.min(), out_fea2.min())
            kl_id_norm = _scale(in_kl, target_min, target_max)  # / kl_id.mean()
            kl_ood_norm = _scale(out_kl, target_min, target_max)  # kl_ood / kl_id.mean()

            bestscore = -10
            best_rec = []

            for beta in range(0, 100, 1):  # in_score + 1.5*
                beta = beta / 10 - 5
                id_score = beta * (-np.min(kl_id_norm, axis=1)) + (-np.max(in_fea2, axis=1))
                ood_score = beta * (-np.min(kl_ood_norm, axis=1)) + (-np.max(out_fea2, axis=1))

                auroc, _, fpr = self.get_measures(-id_score, -ood_score)
                score = auroc - fpr
                if score > bestscore:
                    bestscore = score
                    best_rec = [beta, auroc, fpr]
                print(ood_name[i], 'beta, auroc, fpr', round(beta, 5), round(auroc, 5), round(fpr, 5))
            print(ood_name[i], 'MAX beta, auroc, fpr', round(best_rec[0], 5), round(best_rec[1], 5),
                  round(best_rec[2], 5))
            mean_auroc += best_rec[2] / len(ood_name)

        return 1 - mean_auroc

    def get_measures(self, _pos, _neg, recall_level=0.95):
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

        return auroc, aupr, fpr

    def get_feature(self, train_loader, loader):
        labels = []
        total_logit1 = []
        total_logit2 = []
        total_logit3 = []
        if type(self.class_sim_mean) == type(None):
            print('get_cls_sim_mean......')
            total_logi = []
            unique_labels = torch.tensor([]).long()
            with torch.no_grad():
                for batch_idx, (images, label) in tqdm(enumerate(train_loader)):
                    images = images.cuda()
                    logits1, logits2, logits3 = self.model.evaluate(images)  # self.model.module.evaluate(images)
                    labels.append(label)
                    total_logi.append(logits2)
    
                    unique_labels = torch.unique(torch.cat([unique_labels, label]))
                    if len(unique_labels) >= 1000 and batch_idx > 200:
                        break
            total_logi = torch.cat(total_logi, dim=0).cpu()
            total_logi = F.softmax(total_logi / 5, dim=-1)
            labels = torch.cat(labels, dim=0).cpu()
            with torch.no_grad():
                class_sim_mean = np.array(
                    [total_logi[(labels == i)].mean(axis=0) for i in range(1000)])
                self.class_sim_mean = class_sim_mean
        labels = []
        bs = 200
        print('get_logits......')
        with torch.no_grad():
            for batch_idx, (images, label) in tqdm(enumerate(loader)):
                images = images.cuda()
                logits1, logits2, logits3 = self.model.evaluate(images)  # self.model.module.evaluate(images)
                labels.append(label)
                total_logit1.append(logits1)
                total_logit2.append(logits2)
                total_logit3.append(logits3)
        total_logit1 = torch.cat(total_logit1, dim=0)
        total_logit2 = torch.cat(total_logit2, dim=0)
        total_logit3 = torch.cat(total_logit3, dim=0)
    
        total_sim = F.softmax(torch.tensor(total_logit2 / 5).float(), dim=-1)
        kl_div = []
        with torch.no_grad():
            print('computing kl.......',)
            for i in tqdm(range(total_sim.shape[0] // bs)):
                cur_logits = total_sim[i * bs: (i + 1) * bs]
                if i == total_sim.shape[0] // bs - 1:
                    cur_logits = total_sim[i * bs:]
                output = cur_logits.data.cpu().numpy()
                kl_div.append(get_kl(self.class_sim_mean, output))
        kl_div = np.concatenate(kl_div, axis=0)
        return total_logit1.detach().cpu().numpy(), total_logit2.detach().cpu().numpy(), total_logit3.detach().cpu().numpy(), kl_div
    

class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS  # 64
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT  # .strip()  #template
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT  # .strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:  # default
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")
        print("Number of Class:", len(classnames))

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls  # 总类别数量
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):  # prompt_gen=None,
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:  # suffix [10, 12, 512]  prefix [10,1,512]
            prefix_pos = self.token_prefix_pos[cls_id]  # [1,2,1,512]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]  # [1,2,12,512]
            suffix_neg = self.token_suffix_neg[cls_id]

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim) # [2,1,64,512]
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = prompts_pos  # torch.cat([prompts_neg, prompts_pos], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts = tokenized_prompts_pos  # torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts[self.n_cls:]

        return prompts, tokenized_prompts


class Proj1(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(Proj1, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, token_embed_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, D) 512
        '''
        x = self.prompt_proj(x.float())
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Proj2(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(Proj2, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, visual_dim, 1),
            nn.ReLU(),
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, token_embed_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, 49, D)
        '''
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, D, 49)
        x = self.prompt_proj(x.float())
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, 49, D)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


