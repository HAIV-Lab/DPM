import argparse
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import warnings
import torchvision
warnings.filterwarnings("ignore")
import datasets.cifar
import trainer.DPM_cifar

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.DPM = CN()
    cfg.TRAINER.DPM.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP_MLC = CN()
    cfg.TRAINER.COOP_MLC.N_CTX_POS = 16
    cfg.TRAINER.COOP_MLC.N_CTX_NEG = 16
    cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = None  # 全部使用可学习的
    cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = None
    cfg.TRAINER.COOP_MLC.CSC  = True

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def set_ood_loader(out_dataset, preprocess, root):

    from torchvision import datasets


    if out_dataset == 'ImageNetr':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenet-r'), transform=preprocess)
    elif out_dataset == 'cifar10':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'cifar10','test'), transform=preprocess)
    elif out_dataset == 'LSUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN_C'), transform=preprocess)
    elif out_dataset == 'LSUN_R':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN_R'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=256,
                                            shuffle=False, num_workers=4)
    return testloaderOut

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    trainer = build_trainer(cfg)
    out_datasets = ['cifar10','ImageNetr','LSUN' ]
    import clip
    model, preprocess = clip.load('ViT-B/16')
    trainset = torchvision.datasets.ImageFolder(os.path.join(args.root,'OOD', 'cifar100', 'train'), transform=preprocess)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                                shuffle=True, num_workers=4)
    testset = torchvision.datasets.ImageFolder(os.path.join(args.root,'OOD', 'cifar100', 'test'), transform=preprocess)
    id_loader = torch.utils.data.DataLoader(testset, batch_size=512,
                                                shuffle=False, num_workers=4)
    ood_loader_list = []
    for out_dataset in out_datasets:
        ood_loader = set_ood_loader(out_dataset, preprocess, root=os.path.join(args.root, 'OOD'))
        ood_loader_list.append(ood_loader)
    trainer.train(train_loader, id_loader, ood_loader_list)
    trainer.test_ood(train_loader, id_loader, ood_loader_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='', help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="./output/test", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1555, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="./configs/trainer/cifar.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="./configs/datasets/cifar100.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="DPM", choices=["DPM"],help="name of trainer")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()

    main(args)