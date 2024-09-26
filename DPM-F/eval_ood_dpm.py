import argparse
import numpy as np
import torch
from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.dpm_core import get_and_print_results, get_ood_scores_DPM, get_sim_mean_DPM ,get_ood_scores_CLIP
from utils.train_eval_util import  set_val_loader, set_ood_loader_ImageNet
import torch.nn.functional as F
import clip
from scipy.stats import entropy

def str2bool(str):
    return True if str.lower() == 'true' else False

def get_text(args, test_labels, net):
    with open("./utils/prompt.txt") as f:
        prompt_lis = f.readlines()
        num_prom = len(prompt_lis)
    text_yes_ttl = torch.zeros(len(test_labels), 512).cuda()
    with torch.no_grad():
        for idx in range(num_prom):  # +aftfix[name]
            text_inputs = torch.cat([clip.tokenize(prompt_lis[idx].replace("\n", "").format(name)[1:-2]) for name in test_labels]).cuda()
            text_features = net.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            text_yes_ttl += text_features
    return F.normalize(text_yes_ttl, dim=-1)  # len(test_labels), 512

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates DPM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default='./data/', type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=1556, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int,
                        help='the GPU indice to use')

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=10,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='DPM', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'DPM'], help='score options')
    parser.add_argument('--gamma', default=0.1, type=float, help="feature aggregation")
    args = parser.parse_args()
    args.n_cls = get_num_cls(args)

    return args

def softmax(x):
    # 计算指数值
    exp_values = np.exp(x )
    # 计算每个样本的Softmax概率
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def main():
    args = process_args()
    print(args)
    print(args.CLIP_ckpt)
    setup_seed(args.seed)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    model, preprocess = clip.load(args.CLIP_ckpt)
    model.eval()


    if args.in_dataset in ['ImageNet']:
         out_datasets = [ 'iNaturalist','SUN', 'places365','dtd']

    test_loader,train_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)


    text_f_yes = get_text(args, test_labels, model)

    if args.score == 'DPM':
        id_sim_mean, id_sim_lables = get_sim_mean_DPM(args, model, train_loader, text_f_yes, test_labels,
                                                        args.gamma)
        ca_id_score, kl_id = get_ood_scores_DPM(args, model, test_loader, text_f_yes, id_sim_mean,args.gamma)
        feature = {}
        feature['ca_id'] = ca_id_score
        feature['kl_id'] = kl_id
        t = args.T
        ca_id_score = np.array([softmax(item / t) for item in ca_id_score])
        VM_AUR,VM_FPR = [], []
        TM_AUR, TM_FPR = [], []
        DPM_AUR, DPM_FPR = [], []
        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
            ca_ood_score, kl_ood = get_ood_scores_DPM(args, model, ood_loader, text_f_yes, id_sim_mean,args.gamma)
            def _scale(x, target_min, target_max):
                y = (x - x.min()) / (x.max() - x.min())
                y *= target_max - target_min
                y += target_min
                return y

            ca_ood_score  = np.array([softmax(item / t) for item in ca_ood_score])
            target_max, target_min = ca_id_score.max(), ca_id_score.min()
            kl_id_norm = _scale(kl_id, target_min, target_max)  # / kl_id.mean()
            kl_ood_norm = _scale(kl_ood, target_min, target_max)  # kl_ood / kl_id.mean()

            bestscore = -10  #search
            for beta in range(0, 50, 1):
                beta = beta / 5 - 5
                id_score = beta * (-np.min(kl_id_norm, axis=1)) + (-np.max(ca_id_score, axis=1))
                ood_score = beta * (-np.min(kl_ood_norm, axis=1)) + (-np.max(ca_ood_score, axis=1))
                auroc, fpr = get_and_print_results( id_score, ood_score)
                score = auroc - fpr

                if score > bestscore:
                    bestscore = score
                    bestaur = auroc
                    bestfpr = fpr
                    bestbeta = beta
                    # print('beta,auroc,fpr', beta, auroc, fpr)

            print('******************dpm**********************')
            print('bestbeta,auroc,fpr',bestbeta, bestaur, bestfpr)
            DPM_AUR.append(bestaur)
            DPM_FPR.append(bestfpr)

            print('******************vm**********************')
            id_score = (np.min(kl_id_norm, axis=1))
            ood_score = (np.min(kl_ood_norm, axis=1))

            auroc, fpr = get_and_print_results( id_score, ood_score)
            print('VM,auroc,fpr',  auroc, fpr)
            VM_AUR.append(auroc)
            VM_FPR.append(fpr)

            print('******************TM**********************')
            id_score = (-np.max(ca_id_score, axis=1))
            ood_score =  (-np.max(ca_ood_score, axis=1))

            auroc, fpr = get_and_print_results( id_score, ood_score)
            print('TM,auroc,fpr',  auroc, fpr)
            TM_AUR.append(auroc)
            TM_FPR.append(fpr)

        print('******************mean***************')
        print('TM,auroc,fpr', sum(TM_AUR) / len(TM_AUR),sum(TM_FPR) / len(TM_FPR) )
        print('VM,auroc,fpr', sum(VM_AUR) / len(VM_AUR), sum(VM_FPR) / len(VM_FPR))
        print('DPM,auroc,fpr', sum(DPM_AUR) / len(DPM_AUR), sum(DPM_FPR) / len(DPM_FPR))
    else:
        in_score = get_ood_scores_CLIP( model, test_loader, text_f_yes)
        for out_dataset in out_datasets:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=args.root_dir)
            out_score = get_ood_scores_CLIP( model, ood_loader, text_f_yes)
            search(args, in_score, out_score)


def search(args,id,ood,softmax=True,search=True):
    print(args.score,softmax)
    if search==False:
        t=1
        id_score = oodmethod(args, t, id, softmax=softmax)
        ood_score = oodmethod(args, t, ood, softmax=softmax)
        auroc, fpr = get_and_print_results( id_score, ood_score)
        print('t,auroc,fpr:%s,%s,%s', t,  str(auroc), str(fpr))
    else:
        for t in [1,2,5,10,100,1000]:
            id_score = oodmethod(args,t,id,softmax=softmax)
            ood_score = oodmethod(args,t, ood, softmax=softmax)
            auroc, fpr = get_and_print_results( id_score, ood_score)
            print('t,auroc,fpr:%s,%s,%s', t,  str(auroc), str(fpr))

def oodmethod(args,t,logits,softmax=True):
    _score = []
    to_np = lambda x: x.data.cpu().numpy()
    for output in logits:
        output = torch.unsqueeze(output,dim=0)
        if softmax:
            smax = to_np(F.softmax(output / t, dim=1))
        else:
            smax = to_np(output / t)
        if args.score == 'energy':
            _score.append(-to_np((t * torch.logsumexp(output / t,
                                                           dim=1))))  # energy score is expected to be smaller for ID
        elif args.score == 'entropy':
            _score.append(entropy(smax, axis=1))
        elif args.score == 'var':
            _score.append(-np.var(smax, axis=1))
        elif args.score == 'MCM':
            _score.append(-np.max(smax, axis=1))

    return np.array(_score)

if __name__ == '__main__':
    main()