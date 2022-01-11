import our_method.nn_phi as ourphi
import our_method.nn_psi as ourpsi
import our_method.nn_theta as ourth
import our_method.data_helper as ourd
import torch
import torch.utils.data as data_utils
import utils.common_utils as cu
import numpy as np


def assess_th_phi_psi(dh:ourd.DataHelper, nnth:ourth.NNthHelper, nnphi:ourphi.NNPhiHelper, 
                        nnpsi:ourpsi.NNPsiHelper,
                        loader:data_utils.DataLoader=None, *args, **kwargs):
    """
    Returns:
        raw_acc
        rec_acc
        rs
        predicted_betas in order
    """
    cu.set_seed(42)
    if loader is None:
        loader = nnth._tst_loader
    
    nnth._model.eval()
    nnphi._phimodel.eval()
    nnpsi._psimodel.eval()


    raw_accs = []
    rec_accs = []
    rs = []
    pred_betas = []
    with torch.no_grad():
        for _, _, x, y, z, b in loader:
            x, y, z, b = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), z.to(cu.get_device()), b.to(cu.get_device())

            # need recourse
            r = nnpsi._psimodel.forward_r(x, b)
            need_rec = (r > 0.5) * 1

            # predict beta
            pred_b = nnphi._phimodel.forward(x, b)
            rec_x = torch.multiply(z, pred_b)
            rec_x = torch.multiply(rec_x, need_rec.view(-1, 1)) + torch.multiply(x, 1-need_rec.view(-1, 1))

            # raw_acc
            raw_acc = nnth.accuracy(x.cpu().numpy(), y.cpu().numpy())
            rec_acc = nnth.accuracy(rec_x.cpu().numpy(), y.cpu().numpy())

            raw_accs.append(raw_acc)
            rec_accs.append(rec_acc)
            rs.append(need_rec.cpu().numpy())
            pred_betas.append(pred_b.cpu().numpy())

    rs = np.array(rs).flatten()
    pred_betas = np.array(pred_betas).reshape(-1, 10)
    return np.mean(raw_accs), np.mean(rec_accs), rs, pred_betas


def assess_th_phi(dh:ourd.DataHelper, nnth:ourth.NNthHelper, nnphi:ourphi.NNPhiHelper, 
                        loader:data_utils.DataLoader=None, *args, **kwargs):
    """
    Returns:
        raw_acc
        rec_acc
        predicted_betas in order
    """
    cu.set_seed(42)
    if loader is None:
        loader = nnth._tst_loader
    
    nnth._model.eval()
    nnphi._phimodel.eval()

    raw_accs = []
    rec_accs = []
    pred_betas = []
    with torch.no_grad():
        for _, _, x, y, z, b in loader:
            x, y, z, b = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), z.to(cu.get_device()), b.to(cu.get_device())

            # predict beta
            pred_b = nnphi._phimodel.forward(x, b)
            rec_x = torch.multiply(z, pred_b)

            # raw_acc
            raw_acc = nnth.accuracy(x.cpu().numpy(), y.cpu().numpy())
            rec_acc = nnth.accuracy(rec_x.cpu().numpy(), y.cpu().numpy())

            raw_accs.append(raw_acc)
            rec_accs.append(rec_acc)
            pred_betas.append(pred_b.cpu().numpy())

    pred_betas = np.array(pred_betas).reshape(-1, 10)
    return np.mean(raw_accs), np.mean(rec_accs), pred_betas



