import os
import torch
import numpy as np
from tqdm import tqdm

import sklearn.metrics as sk
from torchvision import datasets
import torch.nn.functional as F
import torchvision
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist



def truncate_sentence(sentence, tokenizer):
    """
    Truncate a sentence to fit the CLIP max token limit (77 tokens including the
    starting and ending tokens).

    Args:
        sentence(string): The sentence to truncate.
        tokenizer(CLIPTokenizer): Pretrained CLIP tokenizer.
    """

    cur_sentence = sentence
    tokens = tokenizer.encode(cur_sentence)

    if len(tokens) > 77:
        # Skip the starting token, only include 75 tokens
        truncated_tokens = tokens[1:76]
        cur_sentence = tokenizer.decode(truncated_tokens)

        # Recursive call here, because the encode(decode()) can have different result
        return truncate_sentence(cur_sentence, tokenizer)

    else:
        return cur_sentence


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                                      transform=preprocess)
    return testloaderOut



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

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
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


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def input_preprocessing(args, net, images, text_features=None, classifier=None):
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == 'vit-Linear':
        image_features = net(pixel_values=images.float()).last_hidden_state
        image_features = image_features[:, 0, :]
    elif args.model == 'CLIP-Linear':
        image_features = net.encode_image(images).float()
    if classifier:
        outputs = classifier(image_features) / args.T
    else:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        outputs = image_features @ text_features.T / args.T
    pseudo_labels = torch.argmax(outputs.detach(), dim=1)
    loss = criterion(outputs, pseudo_labels)  # loss is NEGATIVE log likelihood
    loss.backward()

    sign_grad = torch.ge(images.grad.data, 0)  # sign of grad 0 (False) or 1 (True)
    sign_grad = (sign_grad.float() - 0.5) * 2  # convert to -1 or 1

    std = (0.26862954, 0.26130258, 0.27577711)  # for CLIP model
    for i in range(3):
        sign_grad[:, i] = sign_grad[:, i] / std[i]

    processed_inputs = images.data - args.noiseMagnitude * sign_grad  # because of nll, here sign_grad is actually: -sign of gradient
    return processed_inputs


def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0] // bs)):
        curr_batch = test_image_class_distribution[i * bs: (i + 1) * bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)
        q = train_image_class_distribution
        q_repeated = torch.cat([q] * bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[i * bs: (i + 1) * bs, :] = kl

    return kl_divs_sim


def kl(p, q):
    p = p.astype(np.float32)+1e-6
    q = q.astype(np.float32)+1e-6
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_kl(id_sim_mean, logits):
    _score = []
    scores = cdist(logits, id_sim_mean, metric=kl)  # Batch * 100cdist
    _score = scores  # -np.max(kl_logits, axis=1)
    return _score.copy()


def get_sim_mean_DPM(args, model, loader, text_f_yes, test_labels, gamma=0.098):

    to_np = lambda x: x.data.cpu().numpy()
    labels = []
    feat_t = F.normalize(text_f_yes, dim=-1).half()
    L, D = feat_t.shape
    tqdm_object = tqdm(loader, total=len(loader))
    logits1 = []
    logits2 = []
    with torch.no_grad():
            for batch_idx, (images, label,index) in enumerate(tqdm_object):
                if batch_idx > 1000:
                    break
                images = images.cuda()
                label = label.cuda()
                features = model.encode_image(images)
                features = features.permute(1, 0, 2)
                features /= features.norm(dim=-1, keepdim=True)
                Fv = features[:, 0, :]
                Fs = features[:, 1:, :]
                A_weight = F.conv1d(Fs.permute(0, 2, 1), feat_t[:, :, None])  # [B, 2L, 49]
                A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
                feat_v_a = A_weight1 @ Fs  # [B, L, C]
                Fv = Fv.unsqueeze(1)
                Fv = Fv.expand(-1, L, -1)
                feat_v = gamma * feat_v_a + Fv # [B, L, C] + Fv
                l1 = torch.mul(Fv, feat_t).sum(-1)
                l2 = torch.mul(feat_v, feat_t).sum(-1)
                logits1.append(l1)#.unsqueeze(0))
                logits2.append(l2)#.unsqueeze(0))  # to_np
                labels.append(label)

            labels = torch.cat(labels, dim=0)
            logits2 = torch.cat(logits2, dim=0)
    total_logits = logits2
    total_sim = F.softmax(total_logits/args.T, dim=1)
    total_sim = to_np(total_sim)
    class_num = len(test_labels)
    class_labels = to_np(labels)
    labels = labels.cpu()
    with torch.no_grad():
        class_sim_mean = np.array(
            [total_sim[(labels == i )].mean(axis=0) for i in range(class_num)])
    print(class_sim_mean)
    return class_sim_mean, class_labels  # class_sim_mean


def get_ood_scores_DPM(args, model, loader, text_f_yes, id_sim_mean, gamma=0.1):
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    feat_t = F.normalize(text_f_yes, dim=-1).half()
    L, D = feat_t.shape
    logits1 = []
    logits2 = []
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(tqdm_object): #tqdm_object
            images = images.cuda()
            features = model.encode_image(images)
            features = features.permute(1, 0, 2)
            features /= features.norm(dim=-1, keepdim=True)
            Fv = features[:, 0, :]
            Fs = features[:, 1:, :]
            A_weight = F.conv1d(Fs.permute(0, 2, 1), feat_t[:, :, None])  # [B, 2L, 49]
            A_weight1 = F.softmax(A_weight, dim=-1)  # [B, 2L, 49]
            feat_v_a = A_weight1 @ Fs  # [B, L, C]
            Fv = Fv.unsqueeze(1)
            Fv = Fv.expand(-1, L, -1)
            feat_v = gamma * feat_v_a + Fv # [B, L, C] + Fv
            l1 = torch.mul(Fv, feat_t).sum(-1)
            l2 = torch.mul(feat_v, feat_t).sum(-1)
            logits1.append(l1)#.unsqueeze(0))
            logits2.append(l2)#.unsqueeze(0))  # to_np
        logits2 = torch.cat(logits2, dim=0)
    total_logits = logits2
    total_sim = F.softmax(total_logits / args.T, dim=1)
    total_sim = to_np(total_sim)
    total_logits = to_np(total_logits / args.T)
    bs = 200
    kl_div = []

    with torch.no_grad():
        for i in range(total_sim.shape[0] // bs):
            cur_logits = total_sim[i * bs: (i + 1) * bs]
            output = cur_logits  # to_np(cur_logits)
            kl_div.append(get_kl(id_sim_mean, output))
        if (i + 1)*bs < total_sim.shape[0]:
            cur_logits = total_sim[(i + 1) * bs:]
            output = cur_logits  # to_np(cur_logits)
            kl_div.append(get_kl(id_sim_mean, output))
    tmp = concat(kl_div)

    return total_logits.copy(), tmp.copy()

def get_ood_scores_CLIP(model, loader, text_f_yes):
    total_features = []
    labels = []
    feat_t = F.normalize(text_f_yes, dim=-1)
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(tqdm_object):
            images = images.cuda()
            label = label.cuda()
            features = model.encode_image(images)
            features = features.permute(1, 0, 2)
            features /= features.norm(dim=-1, keepdim=True)
            total_features.append(features.cpu())
            labels.append(label)
    total_features = torch.cat(total_features, dim=0).cpu()
    img_global_feat = total_features[:, 0, :].cpu()
    logits =  100*img_global_feat.float() @ feat_t.float().cpu().permute(1, 0)
    return logits

def get_and_print_results(in_score, out_score):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    # aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    return measures[0],measures[2]

class TextDataset(torch.utils.data.Dataset):
    '''
    used for MIPC score. wrap up the list of captions as Dataset to enable batch processing
    '''

    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Load data and get label
        X = self.texts[index]
        y = self.labels[index]

        return X, y
