'''
eval.py
a generic test script for most models and dataset defined

Supported models
- ConvLSTM
- PredRNN
- MIM
- PhyDNet
- SimVP
- [Need Repo] Earthformer

Supported datasets
- Moving-MNIST
- Stochastic Moving-MNIST
- SEVIR
- HKO-7

CUDA_VISIBLE_DEVICES=-1 python eval.py -d SEVIR_13_12 -m SIMVP_SEVIR_SIGMOID -f <path/to/checkpoint.pt>
'''

import os
import sys
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from torch import nn
#from torch.utils import tensorboard

from models.simvpp import SimVP
from models.convlstm import ConvLSTM, Encoder, Forecaster, EncoderForecaster
from models.predrnn import PredRNN
from models.phydnet import EncoderRNN, PhyCell, ConvLSTM as ConvLSTM_Phy
#from models.e3dlstm import E3DLSTM_Model
from models.mim import MIM_Model

try:
    from nowcasting.hko_iterator import HKOIterator
except RuntimeError as e:
    print(f'Error preparing HKO-7 dataset: (Please ignore this message if you do not need HKO-7) \n{e}')

from data import dutils

import utilspp as utpp
from utilspp import mae, mse, ssim, psnr, csi, pod, far, vpmse, vpmae, lpips, inception_score, fid, fss, rhd, csi_4, csi_16
from config import *

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def GET_MODEL(config):
    model_type = config['model']
    param = config['param']
    logging.info(f'Model Info: {model_type}. \n {param}')
    if model_type == 'simvp':   
        model = SimVP(**param)
    elif model_type == 'convlstm':
        encoder, forecaster = config['ed']()
        # input shape: (T, B, C, H, W) => need to swap the 
        model = EncoderForecaster(encoder, forecaster, **param)
    elif model_type == 'earthformer':
        sys.path.append('earthformer-minimal/')
        from models.earthformer_model import build_model
        if config['dataset'] == 'sevir':            
            model = build_model(f'earthformer-minimal/cfg_{config["dataset"]}.yaml', (13, 384, 384, 1), (12, 384, 384, 1))
        elif config['dataset'] == 'hko':
            model = build_model(f'earthformer-minimal/cfg_{config["dataset"]}.yaml', (5, 480, 480, 1), (20, 480, 480, 1))
        elif config['dataset'] == 'smmnist':
            model = build_model(f'earthformer-minimal/cfg_{config["dataset"]}.yaml', (10, 64, 64, 1), (10, 64, 64, 1))
        elif config['dataset'] == 'meteonet':
            # TODO 4 or 12???
            model = build_model(f'earthformer-minimal/cfg_{config["dataset"]}.yaml', (12, 256, 256, 1), (12, 256, 256, 1))
        else:
            raise NotImplementedError()
    elif model_type == "predrnn":
        model = PredRNN(num_layers=param['num_layers'], num_hidden=param['num_hidden'], configs=dotdict(param['configs']))
    elif model_type == "phydnet":  
        phycell = PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device='cuda') 
        convcell = ConvLSTM_Phy(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device='cuda')   
        model = EncoderRNN(phycell, convcell) 
    elif model_type == "mim":
        model = MIM_Model(num_layers=param['num_layers'], num_hidden=param['num_hidden'], configs=dotdict(param['configs']))
    else:
        raise Exception(f'Undefined model type: {model_type}')
    return model

def GET_DATALOADER(meta, param, batch_size):
    if meta['dataset'] == 'SEVIR':
        # assume test set at the moment
        return dutils.SEVIRDataIterator(**param, batch_size=batch_size) 
    elif meta['dataset'] == 'HKO-7':
        return HKOIterator(**param)
    elif meta['dataset'] == 'Moving-MNIST':
        _, _, test_loader, _ = dutils.load_mmnist_data(batch_size, batch_size, num_workers=2, **param)
        return iter(test_loader)
    elif meta['dataset'] == 'SMoving-MNIST':
        _, _, test_loader, _, _ = dutils.load_motion_mmnist_data(batch_size, batch_size, num_workers=2, **param)
        return iter(test_loader)
    elif meta['dataset'] == 'KTH':
        #dataset = dutils.KTHActions(**param)  
        #return iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        _, validate_iter = dutils.load_kth_data(batch_size=16, val_batch_size=16, data_root='data/kth_actions', num_workers=2, pre_seq_length=10, aft_seq_length=20)
        return iter(validate_iter)
    elif meta['dataset'] == 'meteo':
        _, valid_loader = dutils.load_MeteoNet_data(batch_size, batch_size, train=False, num_workers=0, **param)
        return iter(valid_loader)
    else:
        raise Exception(f'Undefined dataset config name: {meta["dataset"]}')

class MetricListEvaluator():    
    '''
    To evaluate a list of metrics. Supported metrics:
    - CSI (Eg. `csi-84`)
    - POD (Eg. `pod-84`)
    - FAR (Eg. `far-84`)
    - MAE
    - MSE
    - SSIM
    - PSNR    
    '''
    def __init__(self, metric_list):        
        self.metric_holder = {}
        self.batch_count = 0
        for metric_name in metric_list:
            threshold = ''
            if '-' in metric_name:
                metric_name, threshold = metric_name.split('-')
            # initialize metrics
            key_name = metric_name + (f'-{threshold}' if len(threshold) > 0 else '')
            threshold = float(threshold) / 255 if threshold.isdigit() else threshold
            self.metric_holder[key_name] = self.init_metric(metric_name, threshold=threshold)
    
    def init_metric(self, metric_name, **kwarg):
        '''
        return a tuple of three items in order:
        - the function to call during eval
        - the value(s) to keep track of
        - a dict of any additional item to pass into the function
        '''        
        if metric_name in ['csi', 'pod', 'far']:
            # use tfpn instead
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float32), {'threshold': kwarg['threshold']}] # tp, 
        elif metric_name == 'csi_4':
            # tfpn with radius (pooling)
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float32), {'threshold': kwarg['threshold'], 'radius': 4}]
        elif metric_name == 'csi_16':
            return [utpp.tfpn, np.array([0, 0, 0, 0], dtype=np.float32), {'threshold': kwarg['threshold'], 'radius': 16}]
        else:
            # directly convert the string name into function call
            return [eval(metric_name), 0, {}]

    def eval(self, y_pred, y):
        self.batch_count += 1
        for _, metric in self.metric_holder.items():
            temp = metric[0](y_pred, y, **metric[-1])      
            if temp is list:
                temp = np.array(temp)
            elif type(temp) == torch.Tensor:
                temp = temp.detach().cpu().numpy()
            metric[1] += temp
            
    def get_results(self):
        output_holder = {}
        for key, metric in self.metric_holder.items():
            val = metric[1]
            # special handle of tfpn => compute the final score now
            if metric[0] is utpp.tfpn:
                metric_name, threshold = key.split('-')
                val = eval(metric_name)(*list(metric[1]))
            else:
                val /= self.batch_count if self.batch_count > 0 else 1
            output_holder[key] = val
        return output_holder

# ===============================================================================================
# MAIN
# ===============================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset related
    parser.add_argument('-d', '--dataset', type=str, default='', help='the dataset definition to be set')
    # model related
    parser.add_argument('-f', dest='checkpt', type=str, default='', help='model checkpoint to be loaded from (Empty = not loading)')
    parser.add_argument('-m', '--model', type=str, default='', help='the model definition to be created')
    # hyperparams
    parser.add_argument('-s', '--step', type=int, default=-1, help='The number of steps to run. -1: the entire dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')    
    # config override
    parser.add_argument('--metrics', type=str, default=None, help='An overriding list of metrics to be evaluated, separated by character /')
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    parser.add_argument('-o', '--output', default=None, help='The path to save the log files')
    args = parser.parse_args()

    # prepare logger
    if args.output is None:
        path_list = args.checkpt.split("/")
        logfile_name = os.path.join(*path_list[:-1], 'logs', f'{path_list[-1]}.log')
    else:
        logfile_name = os.path.join(args.output, f'{"_".join(args.checkpt.split("/")[-2:])}.log')
    logging.basicConfig(level=logging.NOTSET, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    logging.info(f'Model checkpoint: {args.checkpt}')
    logging.info(f'Steps: {args.step}')

    # define the config for model and dataset
    model_config = globals()[args.model]
    dataset_config = globals()[args.dataset]
    dataset_param, dataset_meta = dataset_config['param'], dataset_config['meta']

    # prepare dataloader
    seq_len = dataset_meta['seq_len']
    out_len = dataset_meta['out_len']
    loader = GET_DATALOADER(dataset_meta, dataset_param, args.batch_size)

    # prepare model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GET_MODEL(model_config).to(device) 
    model.load_state_dict(torch.load(args.checkpt, map_location=torch.device(device)))

    # prepare metrics
    metric_list = dataset_meta['metrics']
    if args.metrics is not None: 
        metric_list = args.metrics.lower().split('/')
        logging.info(f'Overwriting metrics list with: {metric_list}')

    evaluator = MetricListEvaluator(metric_list)
    step = 1
    while args.step < 0 or step <= args.step:
        model.eval()        

        # load and process data accordingly
        if dataset_meta['dataset'] == 'HKO-7':  
            setattr(args, 'seq_len', seq_len)
            try:
                data = loader.sample(batch_size=args.batch_size)
            except Exception as e:
                logging.error(e)
                break
            x_seq, x_mask, dt_clip, _ = data
            #setattr(args, 'resize', 128) # uncomment this line if you want to reshape
            x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args) 
        elif dataset_meta['dataset'] == 'SEVIR':
            data = loader.sample(batch_size=args.batch_size)
            if data == None: break            
            x, y = data['vil'][:, :seq_len], data['vil'][:, seq_len:]
        else:
            try:
                data = next(loader)
            except Exception as e:
                logging.error(e)
                break   
            if len(data) == 2:
                x, y = data
            else:
                x, y = data[:, :seq_len], data[:, seq_len:]

        with torch.no_grad():
            x = x.to(device)     
            y = y.to(device)            

            # model preprocessing
            if model_config['pre'] is not None:
                x = model_config['pre'](x)
            # inference
            if 'scheduled_sampling' in model_config and model_config['scheduled_sampling']:
                if model_config['pre'] is not None:
                    y_patch = model_config['pre'](y)
                x_y = torch.cat([x, y_patch], dim=1) # concat along dim t
                input_flag = torch.zeros(x_y.shape) # all 0, the model will use all of its prediction to reconstruct              
                #input_flag[:, :seq_len-1] = 1.0
                y_pred = model(x_y, torch.Tensor(input_flag).to(device))[:,seq_len-1:]  
            elif 'reversed_scheduled_sampling' in model_config:
                pass
            else:
                y_pred = model(x) 
            # model postprocessing
            if model_config['post'] is not None:
                y_pred = model_config['post'](y_pred)
            y_pred = torch.clamp(y_pred, 0, 1)

            #utpp.torch_visualize({'x': data[0][0].unsqueeze(0),\
            #                    'gt': data[1][0].unsqueeze(0),\
            #                    'pred': y_pred[0].unsqueeze(0)}, 'gg.png')

        # evaluate the metrics
        evaluator.eval(y_pred, y)

        # log/print every
        if step == 1 or step % args.print_every == 0:
            logging.info(f'{step} Steps evaluated')
            
        step += 1
    

    # log the final scores
    final_results = evaluator.get_results()
    for k, v in final_results.items():
        logging.info(f'{k}: {v}')
