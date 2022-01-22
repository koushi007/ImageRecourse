from pathlib import Path
from torchvision import transforms


SYNTHETIC = "synthetic"
SHAPENET_SAMPLE = "shapenet_sample"
SHAPENET = "shapenet"

SYN_DIR = Path("our_method/data/syn")
SHAPENET_DIR = Path("our_method/data/shapenet")
SHAPENET_DIR_SAI = Path("/mnt/infonas/data/saikoushik")
TB_DIR = Path("tblogs/")
NUMR_PERITER = "num_r_per_iter"

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
OPTIMIZER = "optimizer"
BATCH_NORM = "batch_norm"

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

Y_DICT = {
        0:  "02691156",
        1:  "02828884",
        2:  "02924116",
        3:  "02933112",
        4:  "03001627",
        5:  "03211117",
        6:  "03624134",
        7:  "03636649",
        8:  "03691459", 
        9:  "04090263", 
}

NUM_DICT = {
        0 : 250,
        1 : 70,
        2 : 80  
}

DIST_DICT = {0 : 0.5,
            1 : 1.5,
            2 : 4,
            }
    
SHAPENET_TRAIN_MODEL_PATHS = "/mnt/infonas/data/saikoushik/train_model_path.txt"
SHAPENET_VAL_MODEL_PATHS = "/mnt/infonas/data/saikoushik/val_model_path.txt"
SHAPENET_TEST_MODEL_PATHS = "/mnt/infonas/data/saikoushik/test_model_path.txt"

TEMP_IMG_PATH = "/mnt/infonas/data/saikoushik/temp"

    