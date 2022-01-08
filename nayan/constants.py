from torch.nn.modules.padding import ConstantPad1d
from config_new import config
import numpy as np
from pathlib import Path
import random
import torch
import torchvision.transforms as transforms


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

BRI = "brightness"
SAT = "saturation"
CONT = "contrast"
ROT = "rotation"
TRNSX = "transx"
TRNSY = "transy"
HUE = 'hue'

beta_mean = {
    BRI: 0.,
    SAT: 1.,
    HUE: 0.,
    CONT: 1.,
    ROT: 0.,
    TRNSX: 0.,
    TRNSY: 0.
    }

delta_beta_initial = None
delta_beta_idx = None
delta_beta_order = None
# if config["data"]["beta_type"] == "relative":
#     delta_beta_initial = np.array([0,1,0,1,0,0,0])
#     delta_beta_idx = {
#         BRI: 0,
#         SAT : 1,
#         HUE : 2,
#         CONT : 3,
#         ROT : 4,
#         TRNSX : 5,
#         TRNSY : 6
#     }
#     delta_beta_order = [BRI, SAT, CONT, HUE, ROT, TRNSX, TRNSY]

# elif config["data"]["beta_type"] == "exact":
#     delta_beta_initial = np.array([0,1,1,0,0,0])
#     delta_beta_idx = {
#             BRI: 0,
#             SAT : 1,
#             CONT : 2,
#             ROT : 3,
#             TRNSX : 4,
#             TRNSY : 5
#         }
#     delta_beta_order = [BRI, SAT, CONT, ROT, TRNSX, TRNSY]
    
transxy_scale = 4.
cifar_classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

root_dir = Path("/mnt/infonas/data/nayanakarsh/Algo_Recourse")

random.seed(42)
np.random.seed(42)


device = None
if config["cuda"] != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



reg_tasks = None
# if config["regularizer"]["recourse_type"] == "bin":
#     reg_tasks = {
#         BRI: 4,
#         SAT: 4,
#         CONT: 4,
#         ROT: 6,
#         TRNSX: 4,
#         TRNSY: 4
#     }
# elif config["regularizer"]["recourse_type"] == "inc_dec":
#     reg_tasks = {
#         BRI: 3,
#         SAT: 3,
#         CONT: 3,
#         ROT: 3,
#         TRNSX: 3,
#         TRNSY: 3
#     }

# LOWERS = {
#     BRI: np.array([0, 0.3, 0.5, 0.7]),
#     SAT: np.array([0, 0.3, 0.5, 0.7]),
#     CONT: np.array([0, 0.3, 0.5, 0.7]),
#     ROT: np.array([-180, -60, -20, 0, 20, 60])/180,
#     TRNSX: np.array([-4, -2, 0, 2])/4,
#     TRNSY: np.array([-4, -2, 0, 2])/4
# }
# UPPERS = {
#     BRI: np.array([0.3, 0.5, 0.7, 1]),
#     SAT: np.array([0.3, 0.5, 0.7, 1]),
#     CONT: np.array([0.3, 0.5, 0.7, 1]),
#     ROT: np.array([-60, -20, 0, 20, 60, 180])/180,
#     TRNSX: np.array([-2, 0, 2, 4])/4,
#     TRNSY: np.array([-2, 0, 2, 4])/4
# }

# mobilenet_transform = transforms.Compose([
#             # transforms.ToPILImage(),
#             # transforms.Resize(256),
#             # transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

mobilenet_transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )

mobilenet_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ]
        )

tensor_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

mobilenet_transform_batch = transforms.Compose(
            [
                
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )

mobilenet_inverse_transform = transforms.Compose(
            [
                
                transforms.Normalize((0, 0, 0), (1/0.2471, 1/0.2435, 1/0.2616)),
                transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
            ]
        )

LOG_FORMAT = "[%(levelname)s: %(filename)s:%(funcName)s %(message)s]"