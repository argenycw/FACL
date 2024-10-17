# ============================================================
# utils2.py
# This script contains the utility functions for PyTorch
# Much better refactored and formatted than utils.py
# ============================================================
import os
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import lpips as lp
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchmetrics

from data import dutils

# =======================================================================
# Utils in utils :)
# =======================================================================
def to_cpu_tensor(*args):
    '''
    Input arbitrary number of array/tensors, each will be converted to CPU torch.Tensor
    '''
    out = []
    for tensor in args:
        if type(tensor) is np.ndarray:
            tensor = torch.Tensor(tensor)    
        if type(tensor) is torch.Tensor:
            tensor = tensor.cpu()
        out.append(tensor)
    # single value input: return single value output
    if len(out) == 1:
        return out[0]
    return out

def merge_leading_dims(tensor, n=2):
    '''
    Merge the first N dimension of a tensor
    '''
    return tensor.reshape((-1, *tensor.shape[n:]))

def reshape_patch(img_tensor, patch_size):
    '''
    input shape requirement: (B, T, H, W, C)
    '''    
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = img_tensor.reshape(batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels)
    b = a.transpose(3, 4)
    patch_tensor = b.reshape(batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels)
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    '''
    input shape requirement: (B, T, H, W, C)
    '''
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels)
    b = a.transpose(3, 4)
    img_tensor = b.reshape(batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels)
    return img_tensor

def schedule_sampling(shape, itr, eta, sampling_stop_iter, sampling_changing_rate):
    (b, t, h, w, c) = shape
    t -= 1
    zeros = np.zeros((b, t, h, w, c))

    if itr < sampling_stop_iter:
        eta -= sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample((b, t))
    true_token = (random_flip < eta)
    ones = np.ones((h, w, c))
    zeros = np.zeros((h, w, c))
    real_input_flag = []
    for i in range(b):
        for j in range(t):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (b, t, h, w, c))
    return eta, real_input_flag

def reserve_schedule_sampling_exp(shape, itr, r_sampling_step_1, r_sampling_step_2, r_exp_alpha):
    (b, t, h, w, c) = shape
    if itr < r_sampling_step_1:
        r_eta = 0.5
    elif itr < r_sampling_step_2:
        r_eta = 1.0 - 0.5 * np.exp(-float(itr - r_sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < r_sampling_step_1:
        eta = 0.5
    elif itr < r_sampling_step_2:
        eta = 0.5 - (0.5 / (r_sampling_step_2 - r_sampling_step_1)) * (itr - r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((b, t))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((b, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((h, w, c))
    zeros = np.zeros((h, w, c))
    real_input_flag = []
    for i in range(b):
        for j in range(t - 1):
            if j < t - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (t - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


# =======================================================================
# Loss Functions and Metrics
# =======================================================================

def get_weight(y, b_dict={0.46:2, 0.55:5, 0.62:10, 0.725:30}):
    '''returns a numpy array (no grad) of the ground truth map '''
    w = torch.ones_like(y)
    for k in sorted(b_dict.keys()):
        w[y >= k] = b_dict[k]
    return w 

def hko7_preprocess(x_seq, x_mask, dt_clip, args):
    resize = getattr(args, 'resize', 480)
    seq_len = getattr(args, 'seq_len', 5) #args.seq_len if 'seq_len' in args else 5

    # post-process on HKO-7
    x_seq = x_seq.transpose((1, 0, 2, 3, 4)) / 255. # => (batch_size, seq_length, 1, 480, 480)
    x_seq = dutils.nonlinear_to_linear_batched(x_seq, dt_clip)

    b, t, c, h, w = x_seq.shape
    assert c == 1, f'# channels ({c}) != 1'

    # resize (downsample) the images if necessary
    x_seq = torch.Tensor(x_seq).float().reshape((b*t, c, h, w))
    if resize != h:
        tform = T.Compose([
            T.ToPILImage(), 
            T.Resize(resize),
            T.ToTensor(),
        ])
    else:
        tform = T.Compose([])

    x_seq = torch.stack([tform(x_frame) for x_frame in x_seq], dim=0)
    x_seq = x_seq.reshape((b, t, c, resize, resize))

    x, y = x_seq[:, :seq_len], x_seq[:, seq_len:]
    return x, y

def tfpn(y_pred, y, threshold, radius=1):
    '''
    convert to cpu, and merge the first two dimensions
    '''
    y = merge_leading_dims(y)
    y_pred = merge_leading_dims(y_pred)
    with torch.no_grad():
        if radius > 1:
            pool = nn.MaxPool2d(radius)
            y = pool(y)
            y_pred = pool(y_pred) 
        y = torch.where(y >= threshold, 1, 0)
        y_pred = torch.where(y_pred >= threshold, 1, 0)
        mat = torchmetrics.functional.confusion_matrix(y_pred, y, task='binary', threshold=threshold)
        (tn, fp), (fn, tp) = to_cpu_tensor(mat)
    return tp, tn, fp, fn

def csi(tp, tn, fp, fn):
    '''Critical Success Index. The larger the better.'''
    if (tp + fn + fp) < 1e-7:
        return 0.
    return tp / (tp + fn + fp)

def csi_4(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def csi_16(tp, tn, fp, fn):
    return csi(tp, tn, fp, fn)

def far(tp, tn, fp, fn):
    '''False Alarm Rate. The smaller the better.'''
    if (tp + fp) < 1e-7:
        return 0.   
    return fp / (tp + fp)

def pod(tp, tn, fp, fn):
    '''Probability of Detection (ML: Recall). The larger the better.'''
    if (tp + fn) < 1e-7:
        return 0.    
    return tp / (tp + fn)

def ssim(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    b, t, c, h, w = y.shape
    y = y.reshape((b*t, c, h, w))
    y_pred = y_pred.reshape((b*t, c, h, w))
    # to further ensure any of the input is not negative
    y = torch.clamp(y, 0, 1)
    y_pred = torch.clamp(y_pred, 0, 1)
    return torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)(y_pred, y)

def psnr(y_pred, y):
    y, y_pred = to_cpu_tensor(y, y_pred)
    b, t, c, h, w = y.shape
    y = y.reshape((b*t, c, h, w))
    y_pred = y_pred.reshape((b*t, c, h, w))
    acc_score = 0
    for i in range(b*t):
        acc_score += torchmetrics.PeakSignalNoiseRatio(data_range=1.0)(y_pred[i], y[i]) / (b*t)
    return acc_score

def inception_score(y_pred, y):
    raise NotImplementedError()

def fid(y_pred, y):
    raise NotImplementedError()

GLOBAL_LPIPS_OBJ = None # a static variable
def lpips(y_pred, y, net='vgg'):
    # convert the image range into [-1, 1], assuming the input range to be [0, 1]
    y = merge_leading_dims(y)
    y_pred = merge_leading_dims(y_pred)
    y = (2 * y - 1)
    y_pred = (2 * y_pred - 1)
    global GLOBAL_LPIPS_OBJ
    if GLOBAL_LPIPS_OBJ is None:
        GLOBAL_LPIPS_OBJ = lp.LPIPS(net=net).to(y.device)
    return GLOBAL_LPIPS_OBJ(y_pred, y).mean()

class BMAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BMAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y):
        w = torch.sqrt(get_weight(y))
        return F.l1_loss(w * y_pred, w * y, reduction=self.reduction)

class BMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y):
        w = torch.sqrt(get_weight(y))
        return F.mse_loss(w * y_pred, w * y, reduction=self.reduction)

## Note that MAE/MSE loss is different from metrics
# only use the following in evaluation
def vpmae(pred, true):    
    pred, true = to_cpu_tensor(pred, true)
    pred = pred.numpy()
    true = true.numpy()    
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def vpmse(pred, true):
    pred, true = to_cpu_tensor(pred, true)
    pred = pred.numpy()
    true = true.numpy()
    return np.mean((pred-true)**2,axis=(0,1)).sum()

mae = lambda *args: torch.nn.functional.l1_loss(*args).cpu().detach().numpy()
mse = lambda *args: torch.nn.functional.mse_loss(*args).cpu().detach().numpy()

def patches(y, r, stride):
    if y.dim() == 5:
        y = merge_leading_dims(y)
    assert y.dim() == 4, f'expect input to have 4/5 dimensions but got {y.dim()}. Shape: {y.shape}'
    bt, c, h, w = y.shape
    p = y.unfold(-2, r, stride).unfold(-2, r, stride).reshape((-1, c, r, r))    
    return p

def fss(pred, gt, threshold=0.5, window=5):
    '''
    Fractional Skill Score (FSS) \\
    0 - 1, the higher the better.
    '''
    def pad(x, pad_size):
        return torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size))
    
    def t_patches(y, r, stride):
        b, t, c, h, w = y.shape
        p = y.unfold(-2, r, stride).unfold(-2, r, stride).reshape((b*t, -1, r, r))    
        return p
    stride = window // 2    
    pred = t_patches(pad(pred, stride), window, stride) >= threshold
    gt = t_patches(pad(gt, stride), window, stride) >= threshold
    pred_f = pred.sum(dim=[-1,-2]) / (pred.shape[-1] * pred.shape[-2])
    gt_f = gt.sum(dim=[-1,-2]) / (gt.shape[-1] * gt.shape[-2])
    score = 1 - ((pred_f - gt_f) ** 2).sum(dim=[-1]) / (pred_f ** 2 + gt_f ** 2).sum(dim=[-1])
    score[torch.isnan(score)] = 1.
    score = score.mean()
    return score
    

def rhd(pred, gt, window=5, stride=5, bins=10):
    '''
    Regional Histogram Divergence (RHD)
    0 - inf, the KL divergence of two histogram distributions within a window
    '''
    # get patched historgrams
    def get_patched_histograms(y, window, stride, bins):
        y_p = patches(y, window, stride).detach().cpu()
        counts = torch.zeros((len(y_p), bins))
        for i in range(len(y_p)):
            counts[i,:], _ = torch.histogram(y_p[i].reshape(-1), bins=bins, range=(0, 1))
        return counts
    
    def kl_div(gt, pred, dim=-1, epsilon=1e-5):
        # p log (p/q), where p is gt, q is pred.
        gt += epsilon 
        pred += epsilon            
        gt = gt / gt.sum(dim=dim, keepdim=True)
        pred = pred / pred.sum(dim=dim, keepdim=True)        
        return (gt * torch.log(gt / pred)).sum(dim=-1).mean()

    hist_gt = get_patched_histograms(gt, window, stride, bins)
    hist_pred = get_patched_histograms(pred, window, stride, bins)
    return kl_div(hist_gt, hist_pred, dim=-1)

class L1andL2(nn.Module):
    def __init__(self, l1_ratio=0.5, l2_ratio=0.5):
        super(L1andL2, self).__init__()
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.l1_term = nn.L1Loss()
        self.l2_term = nn.MSELoss()

    def forward(self, pred, true):
        return self.l1_ratio * self.l1_term(pred, true) + self.l2_ratio * self.l2_term(pred, true)


class PatchMSSSIMLoss(nn.Module):
    def __init__(self, radius, stride):
        super(PatchMSSSIMLoss, self).__init__()
        self.radius = radius
        self.stride = stride

    # 计算一维的高斯分布向量
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    # 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
    # 可以设定channel参数拓展为3通道
    def create_window(self, window_size, channel=1):

        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # 计算SSIM
    # 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
    # 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    # 正如前面提到的，上面求期望的操作采用高斯核卷积代替。

    def ssim_loss(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel) # 高斯滤波 求均值
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel) # 求均值

        mu1_sq = mu1.pow(2) # 平方
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq # var(x) = Var(X)=E[X^2]-E[X]^2
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2 # 协方差

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

    def msssim(self, img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.ssim_loss(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if normalize:
            mssim = (mssim + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights
        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1]) #返回所有元素的乘积
        return output

    def patches(self, y, r, stride):
        assert y.dim() == 4
        #bt, c, h, w = y.shape
        unfold = torch.nn.Unfold(kernel_size=(r, r), stride=(stride, stride))
        patches = unfold(y)
        pshape = patches.shape
        patches = patches.permute(0, 2, 1).reshape(-1, pshape[-1], r, r)
        return patches

    def forward(self, y_pred, y):
        #y = merge_leading_dims(y)
        #y_pred = merge_leading_dims(y_pred)
        y_p = self.patches(y, self.radius, self.stride)
        y_pred_p = self.patches(y_pred, self.radius, self.stride)
        loss = 1 - self.msssim(y_p, y_pred_p, normalize=True)

        return loss

class NormalizedAmplitudeLoss(nn.Module):
    def __init__(self, window_size, stride, total_step, term1='fc', mode='mlinear', alpha=-1):
        super(NormalizedAmplitudeLoss, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.total_step = total_step
        self.counts = 0
        self.alpha = alpha
        assert term1 in ['fc', 'ssim', 'msssim']
        self.term1 = term1
        assert mode in ['const', 'sigmoid', 'step', 'mlinear']
        self.weight_mode = mode

    def fc_loss(self, fft_pred, fft_truth):
        # FFTs here must be shifted to the center
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator

    def get_alpha(self, H, W, power=1.5):
        self.counts += 1
        h = np.sqrt(H*W)**1.5 # H * sqrt(H)
        center = 0.6 * self.total_step
        # alpha = h/center
        alpha = np.sqrt(H*W)/center

        if self.weight_mode == 'sigmoid':
            return h/(1+np.exp(-(self.counts-center)*alpha)) # ortho 
        elif self.weight_mode == 'step':
            return 0 if self.counts < center else h
        elif self.weight_mode == 'mlinear':
            return 1E-3 if self.counts < center else (self.counts - center)/(self.total_step - center)*h
        else: # Constant
            return 1E-3

    def sep_fftamp(self, f_pred, f_gt):
        n = f_gt.numel()
        f_gt_fft = torch.fft.fftn(f_gt, dim=[-1,-2], norm='ortho')
        f_pred_fft = torch.fft.fftn(f_pred, dim=[-1,-2], norm='ortho')
        f_pred_amp = f_pred_fft.abs()
        f_gt_amp = f_gt_fft.abs()        
        return f_pred_fft, f_gt_fft
        #mse = torch.nn.functional.mse_loss(f_pred, f_gt)
        #return mse, (2*f_gt*f_pred).mean(), (2*f_pred_amp*f_gt_amp).mean()

    def forward(self, y_pred, y):
        # loss term 1
        # ssim_loss = PatchMSSSIMLoss(self.window_size, self.stride)(y_pred, y)
        h, w = y.shape[-2], y.shape[-1]
        y = merge_leading_dims(y)
        y_pred = merge_leading_dims(y_pred) 
        # loss term 2
        #mse_loss, t_2ff, t_2FF = self.sep_fftamp(y_pred, y)
        #freq_loss = mse_loss + torch.abs(t_2ff - t_2FF)
        f_pred_fft, f_gt_fft = self.sep_fftamp(y_pred, y)        
        freq_loss = torch.nn.functional.mse_loss(f_pred_fft.abs(), f_gt_fft.abs())
        if self.term1 == 'fc':
            loss_term1 = self.fc_loss(f_pred_fft, f_gt_fft)
        elif self.term1 == 'ssim':
            loss_term1 = 1.0 - PatchMSSSIMLoss(self.window_size, self.stride).ssim_loss(y_pred, y) 
        elif self.term1 == 'msssim':
            loss_term1 = PatchMSSSIMLoss(self.window_size, self.stride)(y_pred, y)  
        alpha = self.get_alpha(h, w) if self.alpha < 0 else self.alpha
        return loss_term1, alpha * freq_loss

import random
class RandomScheduling(nn.Module):
    def __init__(self, total_step, micro_batch, const_ratio=0.4):
        super(RandomScheduling, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        const_step = int(total_step*const_ratio)
        self.prob_thres = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # dec = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # if(const_ratio!=1):
        #     const = dec[-1] * torch.ones(const_step, device=device)
        # else:
        #     const = torch.zeros(int(total_step*const_ratio), device=device)
        # self.prob_thres = torch.cat((dec, const), dim=0)
        self.micro_batch = micro_batch
        self.step = 0
        self.out = 0

    def get_thres(self):
        if self.step % self.micro_batch == 0:
            prob = self.prob_thres[self.step//self.micro_batch] if self.step//self.micro_batch < len(self.prob_thres) else self.prob_thres[-1]
            self.out = 1 if random.random() > prob else 0
        self.step += 1
        return self.out

    def fcl(self, fft_pred, fft_truth):
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator

    def fal(self, fft_pred, fft_truth):
        return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())

    def cos_loss(self, fft_pred, fft_truth):
        # Cosine Similarity Loss
        numerator = fft_pred.real*fft_truth.real + fft_pred.imag*fft_truth.imag
        denominator = fft_pred.abs()*fft_truth.abs() + 1E-7 
        return (1. - numerator/denominator).mean()

    def forward(self, pred, gt):
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2], norm='ortho')
        # prob = 1 if random.random() > self.prob_thres[self.step] else 0
        prob = self.get_thres()

        _, _, _, H, W = pred.shape
        weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
        loss = loss*weight
        # self.step += 1
        return loss

# =======================================================================
# Data Visualization
# =======================================================================

def torch_visualize(sequences, savedir='', horizontal=10, vmin=0, vmax=1):
    '''
    input: sequences, a list/dict of numpy/torch arrays with shape (B, T, C, H, W) 
    C is assumed to be 1 and squeezed 
    If batch > 1, only the first sequence will be printed 
    '''        
    # First pass: compute the vertical height and convert to proper format
    vertical = 0
    display_texts = []
    if (type(sequences) is dict):
        temp = []
        for k, v in sequences.items():
            vertical += int(np.ceil(v.shape[1] / horizontal)) 
            temp.append(v)
            display_texts.append(k)            
        sequences = temp
    else:
        for i, sequence in enumerate(sequences):
            vertical += int(np.ceil(sequence.shape[1] / horizontal))
            display_texts.append(f'Item {i+1}')
    sequences = to_cpu_tensor(*sequences)
    # Plot the sequences   
    j = 0
    fig, axes = plt.subplots(vertical, horizontal, figsize=(2*horizontal, 2*vertical), tight_layout=True)
    plt.setp(axes, xticks=[], yticks=[])
    for k, sequence in enumerate(sequences):
        # only take the first batch, now seq[0] is the temporal dim
        sequence = sequence[0].squeeze() # (T, H, W)
        axes[j, 0].set_ylabel(display_texts[k], fontsize=16)
        for i, frame in enumerate(sequence):
            j_shift = j + i // horizontal 
            i_shift = i % horizontal
            axes[j_shift, i_shift].imshow(frame, vmin=vmin, vmax=vmax, cmap='gray')
        j += int(np.ceil(sequence.shape[0] / horizontal))    
    if savedir:
        plt.savefig(savedir + '' if len(savedir)>0 else 'out.png')
        plt.close()
    else:
        plt.show()   


# =======================================================================
# Model Preparation, saving & loading (copied from utils.py)
# =======================================================================
def build_model_name(model_type, model_config):
    '''
    Build the model name (without extension)
    '''
    model_name = model_type + '_'
    for k, v in model_config.items():
        model_name += k
        if type(v) is list or type(v) is tuple:
            model_name += '-'
            for i, item in enumerate(v):
                model_name += (str(item) if type(item) is not bool else '') + ('-' if i < len(v)-1 else '')                
        else:
            model_name += (('-' + str(v)) if type(v) is not bool else '')
        model_name += '_'
    return model_name[:-1]

def extract_model_name(name):
    '''
    An extractable model name should be:
    <model type>_<param2>-<v1>-<v2>_<param2>-<v1>-<v2>.ckpt
    '''
    # if name is a path => look at the basename
    name = os.path.splitext(os.path.basename(os.path.normpath(name)))[0]
    param_dict = {}
    model_param_list = name.strip().split('_')
    model_type = model_param_list[0].lower()
    for param_list in model_param_list[1:]:
        param_list_splitted = param_list.split('-')
        if len(param_list_splitted) > 1:
            param_dict[param_list_splitted[0]] = [eval(i) for i in param_list_splitted[1:]]
        else:
            param_dict[param_list_splitted[0]] = True
    return model_type, param_dict

def build_model_path(base_dir, dataset_type, model_type, timestamp=None):
    if timestamp is None:
        return os.path.join(base_dir, dataset_type, model_type)
    elif timestamp == True:
        return os.path.join(base_dir, dataset_type, model_type, pd.Timestamp.now().strftime('%Y%m%d%H%M%S'))
    return os.path.join(base_dir, dataset_type, model_type, timestamp)


