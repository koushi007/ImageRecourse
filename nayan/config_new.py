# import constants
# Avoid circular imports
BRI = "brightness"
SAT = "saturation"
CONT = "contrast"
ROT = "rotation"
TRNSX = "transx"
TRNSY = "transy"
HUE = 'hue'

config = {

     "phi": {
        "layers": [256, 128, 64, 10],
        "dropouts": [1., 1., 0.3, 0.1, 1.], # the first dropout valus is just after the mobilenet layer
        "lr": 1e-3,
        "opt": "adamw",
        "tblogs": "tblogs/new_approach/phi_all_3_class",
        "dump_path": "models/new_approach/phi_all_3_class",
        "load_path": "models/new_approach/phi_all_3_class",
        "file_prefix": "phi_",
        "epochs": 40,
        "batch_size": 32
    },
    "psi": {
        "layers": [256, 128, 64, 1],
        "dropouts": [1., 1., 0.3, 0.1, 1.], # the first dropout valus is just after the mobilenet layer
        "lr": 1e-3,
        "opt": "adamw",
        "tblogs": "tblogs/new_approach/psi_all_3_class",
        "dump_path": "models/new_approach/psi_all_3_class",
        "load_path": "models/new_approach/psi_all_3_class",
        "file_prefix": "phi_",
        "epochs": 40,
        "batch_size": 32
    },
    "dataset": "cifar",
    "cuda": 0, # pass -1 for CPU
    "logger_file": "psi_and_phi_3_class",
    "plots_dir": "Experiments/Plots"
}