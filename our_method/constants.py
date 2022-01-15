from pathlib import Path
from torchvision import transforms


SYNTHETIC = "synthetic"
SHAPENET_SAMPLE = "shapenet_sample"
SHAPENET = "shapenet"

SYN_DIR = Path("our_method/data/syn")
SHAPENET_DIR = Path("our_method/data/shapenet")
SHAPENET_DIR_SAI = Path("/mnt/infonas/data/saikoushik")
TB_DIR = Path("tblogs/")

LOGREG = "LR"
RESNET = "resnet"
MOBNET_V1 = "mobilenet_v1"
MOBNET_V2 = "mobilenet_v2"

SEQUENTIAL = "baseline"
SEQUENTIAL_KL = "baselinekl"
METHOD1 = "our_method1"
METHOD2 = "our_method2"
RWD_METHOD = "rwd_method"

THETA = "th"
PHI = "phi"
PSI = "psi"

PRETRN_THPSIPSI = "pretrn_th_phi_psi"

LRN_RATTE = "lr"
MOMENTUM = "momentum"

SW = "summarywriter"
BATCH_SIZE = "batch_size"


RESNET_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

TRANSFORM = "transform"

SCHEDULER = "scheduler"
SCHEDULER_TYPE = "scheduler_type"