'''
train_hko7.py
- the training script for any supported model for HKO-10

Sample Command: 

HKO-7:
CUDA_VISIBLE_DEVICES=2 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m CONVLSTM_HKO7 -s 50000 --loss bmse
'''


import os
import torch
import logging
import argparse
import numpy as np

from torch import nn
from torch.utils import tensorboard

from models.simvpp import SimVP
from nowcasting.hko_iterator import HKOIterator
from data import dutils

import utilspp as utpp
from eval import GET_MODEL, MetricListEvaluator
from config import *

try:
    from models.earthformer_model import SequentialLR, warmup_lambda
except ModuleNotFoundError as e:
    print(e)
   
def get_loss(loss, args): 
    if loss == 'mae':
        return nn.L1Loss()
    elif loss == 'mse':
        return nn.MSELoss()
    elif loss.startswith('facl'):
        loss_arglist = loss.split('-')
        return utpp.RandomScheduling(args['step']//args['micro_batch'], args['micro_batch'], const_ratio=float(loss_arglist[-1]))
    else:
        raise Exception(f'Undefined Loss type: {loss}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset related
    parser.add_argument('train_data', type=str, nargs=1, help='the path of the training pickle selection file')
    parser.add_argument('test_data', type=str, nargs=1, help='the path of the test pickle selection file')         
    parser.add_argument('--seq_len', type=int, default=5, help='The input sequence length')
    parser.add_argument('--out_len', type=int, default=20, help='The output (prediction) sequence length') 
    parser.add_argument('--resize', type=int, default=480, help='New size for the frames')
    # model related
    parser.add_argument('-f', type=str, default='', help='model checkpoint to be loaded from (Empty = not loading)')
    parser.add_argument('-o', '--output', type=str, default='checkpoints', help='The output directory')
    parser.add_argument('-m', '--model', type=str, default='', help='The global configuration to be used (The var name in config.py)')
    # hyperparams
    parser.add_argument('--lr', type=float, default=0.001, help='The initial learning rate')
    parser.add_argument('-s', '--step', type=int, default=50000, help='The number of steps to run')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='The batch size') 
    parser.add_argument('--micro_batch', type=int, default=1, help='Micro batch size. (Gradients of N microbatch are accumulated)')   
    parser.add_argument('-l', '--loss', type=str, default='bmse', help='the loss used to train the model (mae, mse, bmae, bmse)')
    parser.add_argument('--scheduler', type=str, default='cosine', help='which lr scheduler to use (cyclic, reduce, cosine)')
    parser.add_argument('--alpha', type=float, default=-1, help='This param has different meaning with different loss')
    parser.add_argument('-w', '--wt', type=str, default='mlinear', help='The function type to set up w (constant, step, mlinear, sigmoid)')
    # logging related
    parser.add_argument('--print_every', type=int, default=100, help='The number of steps to log the training loss')
    parser.add_argument('--validate_every', type=int, default=500, help='The number of steps to perform validation once')
    parser.add_argument('--v_steps', type=int, default=20, help='Validation steps')    
    parser.add_argument('--remarks', type=str, default='', help='This section will affect the model name to be saved')    
    args = parser.parse_args()

    # Actaul step: args.step * args.micro_batch
    #setattr(args, 'step', args.step // args.micro_batch) 

    # args validation
    assert args.model != '', 'You must specify the model config using -m/--model!'

    # read the model config
    dataset_type = 'hko-7'
    dataset_metrics = ['mae', 'mse', 'ssim', 'psnr', 'csi-84', 'csi-185', 'far-84']
    model_config = globals()[args.model]
    model_type =  model_config['model']
    save_path = utpp.build_model_path(args.output, dataset_type, model_type, timestamp=True) + args.remarks

    if 'scheduled_sampling' in model_config:
        eta = 1.0

    # prepare dataloader
    total_seq_len = args.seq_len + args.out_len
    train_hko_iter = HKOIterator(pd_path=args.train_data[0], sample_mode="random", seq_len=total_seq_len, stride=1)
    test_hko_iter = HKOIterator(pd_path=args.test_data[0], sample_mode="sequent", seq_len=total_seq_len, stride=args.seq_len)

    # define the model
    model_param = model_config['param']
    model_pathname = utpp.build_model_name(model_type, model_param)    

    # prepare logger
    os.makedirs(save_path, exist_ok=True)
    logfile_name = os.path.join(save_path, f'_log.log')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(logfile_name), logging.StreamHandler()], format='%(message)s')
    logging.info(f'args: {args}')
    logging.info('The resulting model will be saved as: {}'.format(os.path.join(save_path, model_pathname)))    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GET_MODEL(model_config).to(device) 
    model = model.to(device)

    criterion = get_loss(args.loss, args=vars(args))    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)      

    if args.scheduler == 'reduce':
        if model_type == 'earthformer':
            warmup_iter = 0.2 * args.step
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter//args.micro_batch, min_lr_ratio=0.1))
            reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=100) # SSIM: MAX mode
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, reduce_scheduler], milestones=[warmup_iter])  
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10) # SSIM: MAX mode
    elif args.scheduler == 'cosine':
        if model_type == 'earthformer': 
            warmup_iter = 0.2 * args.step 
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter//args.micro_batch, min_lr_ratio=0.1))
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.step - warmup_iter)//args.micro_batch, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter//args.micro_batch])                 
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.step//args.micro_batch, eta_min=1e-6) 
    else:
        raise Exception(f'Unsupported scheduler type: {args.scheduler}')

    # Writing logs for tensorboard
    log_dir = os.path.join(save_path, 'logs')
    writer = tensorboard.SummaryWriter(log_dir)    

    best_val_loss = 1e10
    optimizer.zero_grad()
    for total_step in range(1, args.step+1):
        model.train()        

        try:
            x_seq, x_mask, dt_clip, _ = train_hko_iter.sample(batch_size=args.batch_size)
            x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args) 
        except ValueError:
            break
        
        x = x.to(device)     
        y = y.to(device)

        if model_config['pre'] != None:
            x = model_config['pre'](x)

        # handle schedule sampling or reversed schedule sampling
        if 'scheduled_sampling' in model_config:
            if model_config['pre'] != None: 
                y_patch = model_config['pre'](y)
            x_y = torch.cat([x, y_patch], dim=1) # concat along dim t
            eta, input_flag = utpp.schedule_sampling(y_patch.shape, itr=total_step, eta=eta, **model_config['scheduled_sampling'])
            writer.add_scalar('eta', eta, global_step=total_step)
            y_pred = model(x_y, torch.Tensor(input_flag).to(device))
            if model_config['post'] != None: 
                y = model_config['post'](x_y[:,1:]) # note: the model will also predict the input frames in this setting                

        elif 'reversed_scheduled_sampling' in model_config:
            pass # TODO
            raise Exception('reversed_scheduled_sampling is not yet implemented')
        else:            
            y_pred = model(x) 

        if model_config['post'] != None:
            y_pred = model_config['post'](y_pred)
        y_pred_ori = y_pred

        # prediction loss
        loss = criterion(y_pred, y)
        if type(loss) is tuple:
            term1, term2 = loss
            loss = term1 + term2
            loss = loss / args.micro_batch
            loss.backward()
        else:
            loss = loss / args.micro_batch
            loss.backward()
        
        if (total_step+1) % args.micro_batch == 0:
            optimizer.step()
            optimizer.zero_grad()
            if args.scheduler == 'cosine':
                scheduler.step()

        # -----------------------------------------------------
        # On Step End
        # -----------------------------------------------------
        # terminal log every {print_every} steps.
        if total_step == 1 or total_step % args.print_every == 0:            
            if 'term1' in vars() or 'term1' in globals():
                logging.info(f'[Step {total_step}] (Min:{y_pred_ori.min():.3}, Max:{y_pred_ori.max():.3}) Term 1: {float(term1):.5}, Term 2: {float(term2):.5}')
            else:
                logging.info(f'[Step {total_step}] (Min:{y_pred_ori.min():.3}, Max:{y_pred_ori.max():.3}) Loss: {float(loss)}')

        # tensorboard logging
        writer.add_scalar('Training Loss', float(loss), global_step=total_step)

        # validate every {validate_every} epochs
        if total_step == 1 or total_step % args.validate_every == 0:  
            model.eval() 
            evaluator = MetricListEvaluator(dataset_metrics)
            rand_step = np.random.randint(0, args.v_steps - 1) 
            rand_batch = np.random.randint(0, args.batch_size)
            test_hko_iter.reset()
            for v_step in range(args.v_steps): # "v_steps" steps to evaluate
                with torch.no_grad():
                    x_seq, x_mask, dt_clip, _ = test_hko_iter.sample(batch_size=args.batch_size)
                    x, y = utpp.hko7_preprocess(x_seq, x_mask, dt_clip, args)
                    x = x.to(device)
                    y = y.to(device)
                    _x, _y = x, y

                    # model preprocessing
                    if model_config['pre'] is not None:
                        x = model_config['pre'](x)
                    # inference
                    if 'scheduled_sampling' in model_config and model_config['scheduled_sampling']:
                        y_patch = y
                        if model_config['pre'] is not None:
                            y_patch = model_config['pre'](y_patch)
                        x_y = torch.cat([x, y_patch], dim=1) # concat along dim t
                        input_flag = torch.zeros(x_y.shape) # since ss "shifts" the indices, we just input all 0
                        y_pred = model(x_y, torch.Tensor(input_flag).to(device))[:,args.seq_len-1:]              
                    elif 'reversed_scheduled_sampling' in model_config:
                        pass
                    else:
                        y_pred = model(x) 
                    # model postprocessing
                    if model_config['post'] is not None:
                        y_pred = model_config['post'](y_pred)

                    y_pred = torch.clamp(y_pred, 0, 1)

                    # evaluate the metrics
                    evaluator.eval(y_pred, y)

                    # save the input/output randomly
                    if v_step == rand_step:
                        out_x, out_y, out_y_pred = _x[rand_batch].unsqueeze(0), _y[rand_batch].unsqueeze(0), y_pred[rand_batch].unsqueeze(0)
            
            # get results from evaluator
            results = evaluator.get_results()

            # determine whether to reduce lr or not (if needed)
            if args.scheduler == 'reduce' and model_type != 'earthformer':
                # for consistency, we use SSIM as the unified parameter to be picked for the best
                scheduler.step(results['ssim'])

            for k, v in results.items():
                # tensorboard logging validation results
                writer.add_scalar(k, float(v), global_step=total_step) 
            writer.add_scalar('Learning Rate', float(optimizer.param_groups[0]['lr']), global_step=total_step)

            # log to terminal in one line
            logging.info(f'[Validation] ' + ' '.join(f'{k}: {v:.3}' for k, v in results.items()))

            # visualize and save model for {validate_every} steps
            utpp.torch_visualize({'input': out_x, 'ground truth': out_y, 'predicted': out_y_pred}, savedir=os.path.join(save_path, f'temp-{total_step}.png'))            
            val_loss = next(iter(results.items()))[-1]
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), os.path.join(save_path, f'{model_pathname}_step-{total_step}.pt'))
                best_val_loss = val_loss

    torch.save(model.state_dict(), os.path.join(save_path, f'{model_pathname}_final.pt'))
