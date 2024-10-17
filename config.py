from data import dutils
import torchvision.transforms as T
# ===============================================================================================
# Global functions for models
# ===============================================================================================
import torch
import utilspp as utpp
from collections import OrderedDict
from models.convlstm import ConvLSTM, Encoder, Forecaster, EncoderForecaster


def GET_DEFAULT_CONVLSTM_ED(batch_size=-1):
    '''
    The batch size doesn't matter. Not used in the model anyway.
    '''
    convlstm_encoder_params = [[
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],[
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
        ]
    ]
    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 7, 5, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), kernel_size=3, stride=1, padding=1),
        ]
    ]
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], out_len=20)
    return encoder, forecaster

def GET_CONVLSTM_ED_MMNIST(batch_size=-1):
    '''
    The batch size doesn't matter. Not used in the model anyway.
    '''
    convlstm_encoder_params = [[
            OrderedDict({'conv1_leaky_1': [1, 8, 5, 2, 2]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],[
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 8, 8), kernel_size=3, stride=1, padding=1),
        ]
    ]
    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 4, 2, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 8, 8), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
        ]
    ]
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], out_len=10)
    return encoder, forecaster

def GET_CONVLSTM_ED_SEVIR(batch_size=-1):
    '''
    The batch size doesn't matter. Not used in the model anyway.
    '''
    convlstm_encoder_params = [[
            OrderedDict({'conv1_leaky_1': [1, 8, 4, 4, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 4, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],
        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 48, 48), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24), kernel_size=3, stride=1, padding=1),
        ]
    ]
    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 6, 4, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 24, 24), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 48, 48), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), kernel_size=3, stride=1, padding=1),
        ]
    ]
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], out_len=12)
    return encoder, forecaster

def GET_CONVLSTM_ED_KTH(batch_size=-1):
    '''
    The batch size doesn't matter. Not used in the model anyway.
    '''
    convlstm_encoder_params = [[
            OrderedDict({'conv1_leaky_1': [1, 8, 5, 2, 2]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],[
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 64, 64), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
        ]
    ]
    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 4, 2, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 64, 64), kernel_size=3, stride=1, padding=1),
        ]
    ]
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], out_len=20)
    return encoder, forecaster

def GET_CONVLSTM_ED_METEO(batch_size=-1):
    '''
    The batch size doesn't matter. Not used in the model anyway.
    '''
    convlstm_encoder_params = [[
            OrderedDict({'conv1_leaky_1': [1, 8, 5, 2, 2]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],[
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 128, 128), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 64, 64), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
        ]
    ]
    convlstm_forecaster_params = [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 4, 2, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 64, 64), kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 128, 128), kernel_size=3, stride=1, padding=1),
        ]
    ]
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], out_len=12)
    return encoder, forecaster

# ===============================================================================================
# Global model configurations
# ===============================================================================================

## UNET
UNET_MMNIST = {
    'model': 'unet',
    'pre': None,
    'post': None, 
    'param': {
        'seq_len': 10,
        'out_len': 10,
        'last_activation': 'none',
    },
}

UNET_MMNIST_SIGMOID = {
    'model': 'unet',
    'pre': None,
    'post': None, 
    'param': {
        'seq_len': 10,
        'out_len': 10,
        'last_activation': 'sigmoid',
    },
}

SMAATUNET_MMNIST = {
    'model': 'smaatunet',
    'pre': None,
    'post': None, 
    'param': {
        'kernels_per_layer': 2,
        'bilinear': True,
        'reduction_ratio': 2, 
        'last_activation': 'none',
    },
}

SMAATUNET_MMNIST_SIGMOID = {
    'model': 'smaatunet',
    'pre': None,
    'post': None, 
    'param': {
        'kernels_per_layer': 2,
        'bilinear': True,
        'reduction_ratio': 2, 
        'last_activation': 'sigmoid',
    },
}

## SimVP v1
SIMVP_SEVIR_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,
    'param': {     
        'shape_in': (13, 1, 384, 384),
        'shape_out': (12, 1, 384, 384),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

SIMVP_SEVIR = {
    'model': 'simvp',
    'pre': None,
    'post': None,    
    'param': {  
        'shape_in': (13, 1, 384, 384),
        'shape_out': (12, 1, 384, 384),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_HKO7 = {
    'model': 'simvp',
    'pre': None,
    'post': None,     
    'param': {  
        'shape_in': (5, 1, 480, 480),
        'shape_out': (20, 1, 480, 480),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_HKO7_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,     
    'param': {      
        'shape_in': (5, 1, 480, 480),
        'shape_out': (20, 1, 480, 480),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

SIMVP_HKO7_IN20_STANDARD = {
    'model': 'simvp',
    'pre': None,
    'post': None,     
    'param': {  
        'shape_in': (20, 1, 480, 480),
        'shape_out': (20, 1, 480, 480),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_HKO7_IN20_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,     
    'param': {      
        'shape_in': (20, 1, 480, 480),
        'shape_out': (20, 1, 480, 480),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

SIMVP_MMNIST = {
    'model': 'simvp',
    'pre': None,
    'post': None,      
    'param': {      
        'shape_in': (10, 1, 64, 64),
        'shape_out': (10, 1, 64, 64),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_MMNIST_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,    
    'param': {      
        'shape_in': (10, 1, 64, 64),
        'shape_out': (10, 1, 64, 64),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

SIMVP_KTH = {
    'model': 'simvp',
    'pre': None,
    'post': None,    
    'param': {      
        'shape_in': (10, 1, 128, 128),
        'shape_out': (20, 1, 128, 128),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_KTH_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,    
    'param': {      
        'shape_in': (10, 1, 128, 128),
        'shape_out': (20, 1, 128, 128),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 4,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

SIMVP_METEO = {
    'model': 'simvp',
    'pre': None,
    'post': None,
    'param': {     
        'shape_in': (4, 1, 256, 256),
        'shape_out': (12, 1, 256, 256),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'none',
    }
}

SIMVP_METEO_SIGMOID = {
    'model': 'simvp',
    'pre': None,
    'post': None,
    'param': {     
        'shape_in': (4, 1, 256, 256),
        'shape_out': (12, 1, 256, 256),
        'hid_S': 16,
        'hid_T': 256,
        'N_S': 4,
        'N_T': 8,
        'groups': 8,
        'last_activation': 'sigmoid',
    }
}

## ConvLSTM (native)
CONVLSTM_SEVIR = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_SEVIR,
    'param': {
        'last_activation': 'none',
    },
}

CONVLSTM_SEVIR_SIGMOID = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_SEVIR,
    'param': {
        'last_activation': 'sigmoid',
    },
}

CONVLSTM_HKO7 = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # TNCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> TNCHW
    'ed': GET_DEFAULT_CONVLSTM_ED, 
    'param': {
        'last_activation': 'none',
    },
}

CONVLSTM_HKO7_SIGMOID = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # TNCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> TNCHW
    'ed': GET_DEFAULT_CONVLSTM_ED,     
    'param': {
        'last_activation': 'sigmoid',
    },
}

CONVLSTM_MMNIST = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_MMNIST,    
    'param': {
        'last_activation': 'none',
    },
}

CONVLSTM_MMNIST_SIGMOID = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_MMNIST,    
    'param': {
        'last_activation': 'sigmoid',
    },
}

CONVLSTM_KTH = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_KTH,    
    'param': {
        'last_activation': 'none',
    },
}

CONVLSTM_KTH_SIGMOID = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_KTH,    
    'param': {
        'last_activation': 'sigmoid',
    },
}

## ConvLSTM (OpenSTL)
CONVLSTM_MMNIST_OPENSTL = {
    'model': 'openstl_convlstm',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4), # NTCHW -> NTHWC
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3)[:, 9:], # NTHWC -> NTCHW
    'num_hidden': '128,128,128,128',
    'param': {        
        'filter_size': 5,
        'stride': 1,
        'patch_size': 4,
        'layer_norm': 0,
        'reverse_scheduled_sampling': 0,
        'r_sampling_step_1': 25000,
        'r_sampling_step_2': 50000,
        'r_exp_alpha': 5000,
        'in_shape': (10, 1, 64, 64),
        'pre_seq_length': 10,
        'aft_seq_length': 10,
    },
}


## Earthformer
EARTHFORMER_MMNIST = {
    'model': 'earthformer',
    'dataset': 'smmnist',    
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: x.permute(0, 1, 4, 2, 3), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_MMNIST_SIGMOID = {
    'model': 'earthformer',
    'dataset': 'smmnist',
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: torch.sigmoid(x.permute(0, 1, 4, 2, 3)), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_SEVIR = {
    'model': 'earthformer',
    'dataset': 'sevir',    
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: x.permute(0, 1, 4, 2, 3), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_SEVIR_SIGMOID = {
    'model': 'earthformer',
    'dataset': 'sevir',    
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: torch.sigmoid(x.permute(0, 1, 4, 2, 3)), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_HKO7= {
    'model': 'earthformer',
    'dataset': 'hko',
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: x.permute(0, 1, 4, 2, 3), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_HKO7_SIGMOID = {
    'model': 'earthformer',
    'dataset': 'hko',
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: torch.sigmoid(x.permute(0, 1, 4, 2, 3)), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_METEO = {
    'model': 'earthformer',
    'dataset': 'meteonet',
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: x.permute(0, 1, 4, 2, 3), # NTHWC -> NTCHW
    'param': {},
}

EARTHFORMER_METEO_SIGMOID = {
    'model': 'earthformer',
    'dataset': 'meteonet',
    'pre': lambda x: x.permute(0, 1, 3, 4, 2), # NTCHW -> NTHWC
    'post': lambda x: torch.sigmoid(x.permute(0, 1, 4, 2, 3)), # NTHWC -> NTCHW
    'param': {},
}

## PredRNN
PREDRNN_MMNIST = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3), 
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 10,
            'total_length': 20,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 64,                        
            'img_height': 64, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
        }
    },
}

PREDRNN_MMNIST_SIGMOID = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3),
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 10,
            'total_length': 20,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 64,                        
            'img_height': 64, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': 'sigmoid',
        },
    },
}

PREDRNN_SEVIR = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3), 
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 13,
            'total_length': 25,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 384,                        
            'img_height': 384, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': 'sigmoid',
        }
    },
}

PREDRNN_MMNIST_SIGMOID = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3),
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 10,
            'total_length': 20,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 64,                        
            'img_height': 64, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': 'sigmoid',
        },
    },
}

PREDRNN_HKO7 = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3), 
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 5,
            'total_length': 25,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 128,                        
            'img_height': 128, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
        }
    },
}

PREDRNN_HKO7_SIGMOID = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3), 
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 5,
            'total_length': 25,
            'img_channel': 1,
            'patch_size': 4,
            'img_width': 128,                        
            'img_height': 128, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': 'sigmoid',
        }
    },
}

PREDRNN_METEO = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(), # Change Here
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3),  # Change Here
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 4,
            'total_length': 16,
            'img_channel': 1,
            'patch_size': 4, # Change HERE
            'img_width': 128,                        
            'img_height': 128, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': None,
        }
    },
}

PREDRNN_METEO_SIGMOID = {
    'model': 'predrnn',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(), # Change Here
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3),  # Change Here
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'input_length': 4,
            'total_length': 16,
            'img_channel': 1,
            'patch_size': 4, # Change HERE
            'img_width': 128,                        
            'img_height': 128, 
            'filter_size': 5,
            'stride': 1,
            'layer_norm': 1,
            'activation': 'sigmoid',
        }
    },
}

##MIM
MIM_MMNIST = {
    'model': 'mim',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3).contiguous(),
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'pre_seq_length': 10,
            'aft_seq_length': 10,
            'in_shape': [20, 1, 64, 64],
            'filter_size': 5,
            'stride': 1,
            'patch_size': 4,
            'layer_norm': 0,
            'device': 'cuda',
        }
    },
}

MIM_MMNIST_SIGMOID = {
    'model': 'mim',
    'pre': lambda x: utpp.reshape_patch(x.permute(0, 1, 3, 4, 2), 4).contiguous(),
    'post': lambda x: utpp.reshape_patch_back(x, 4).permute(0, 1, 4, 2, 3).contiguous(),
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {        
        'num_layers': 4,
        'num_hidden': [64, 64, 64, 64],
        'configs': {
            'pre_seq_length': 10,
            'aft_seq_length': 10,
            'in_shape': [20, 1, 64, 64],
            'filter_size': 5,
            'stride': 1,
            'patch_size': 4,
            'layer_norm': 0,
            'device': 'cuda',
            'activation': 'sigmoid',
        }
    },
}

##PhyDNet
PHYDNET_MMNIST = {
    'model': 'phydnet',
    'pre': None,
    'post': lambda x: x, 
    'scheduled_sampling': {
        'sampling_stop_iter': 50000,
        'sampling_changing_rate': 0.00002,
    },
    'param': {
        'configs': {
            'patch_size': 1,
        
            'in_shape': [10, 1, 64, 64],
            'pre_seq_length': 10,
            'aft_seq_length': 10,
            'total_length': 20,

            'device': 'cuda',
            'activation': None,
        }
    },
}

## E3DLSTM
E3DLSTM_MMNIST = {
    'model': 'e3dlstm',
    'pre': None,
    'post': None, 
    'param': {
        'reverse_scheduled_sampling': True,
        
        'num_hidden': '128,128,128,128',
        'filter_size': 5,
        'stride': 1,
        'patch_size': 4,
        'layer_norm': 0,

        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,

        'device': 'cuda',
    },
}

CONVLSTM_METEO_256 = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> TNCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # TNCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_METEO,
    'param': {
        'last_activation': 'none',
    },
}

CONVLSTM_METEO_256_SIGMOID = {
    'model': 'convlstm',
    'pre': lambda x: x.permute(1, 0, 2, 3, 4), # NTCHW -> TNCHW
    'post': lambda x: x.permute(1, 0, 2, 3, 4), # TNCHW -> NTCHW
    'ed': GET_CONVLSTM_ED_METEO,
    'param': {
        'last_activation': 'sigmoid',
    },
}

# ===============================================================================================
# Global dataset configurations
# ===============================================================================================

SEVIR_13_12 = {
    'meta': {
        'dataset': 'SEVIR',    
        'seq_len': 13,
        'out_len': 12,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips', 
                    'csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219', 
                    'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219',
                    'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219',
                    'fss', 'rhd'],
    },
    'param': {        
        'seq_len': 25,
        'data_types': ['vil'], 
        'sample_mode': 'sequent',
        'layout': 'NTCHW', 
        'raw_seq_len': 25, 
        'start_date': dutils.SEVIR_TRAIN_TEST_SPLIT_DATE,
        'end_date': None,
    },
}

SEVIR_13_12_POOLING = {
    'meta': {
        'dataset': 'SEVIR',    
        'seq_len': 13,
        'out_len': 12,
        'metrics': ['csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219', 
                    'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219',
                    'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219'],
    },
    'param': {        
        'seq_len': 25,
        'data_types': ['vil'], 
        'sample_mode': 'sequent',
        'layout': 'NTCHW', 
        'raw_seq_len': 25, 
        'start_date': dutils.SEVIR_TRAIN_TEST_SPLIT_DATE,
        'end_date': None,
    },
}

HKO7_5_20 = {
    'meta': {
        'dataset': 'HKO-7',
        'seq_len': 5,
        'out_len': 20,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'csi-84', 'csi-117', 'csi-140', 'csi-158', 'csi-185', 
                    'csi_4-84', 'csi_4-117', 'csi_4-140', 'csi_4-158', 'csi_4-185', 
                    'csi_16-84', 'csi_16-117', 'csi_16-140', 'csi_16-158', 'csi_16-185',
                    'pod-84', 'far-84', 'lpips', 'fss', 'rhd']
    },
    'param': {            
        'pd_path': 'data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl',
        'sample_mode': 'sequent',
        'seq_len': 25,
        'stride': 5,    
    }
}

HKO7_5_20_POOLING = {
    'meta': {
        'dataset': 'HKO-7',
        'seq_len': 5,
        'out_len': 20,
        'metrics': ['mae', 'mse', 'csi-84', 'csi_4-84', 'csi_4-117', 'csi_4-140', 'csi_4-158', 'csi_4-185', 
                    'csi_16-84', 'csi_16-117', 'csi_16-140', 'csi_16-158', 'csi_16-185']
    },
    'param': {            
        'pd_path': 'data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl',
        'sample_mode': 'sequent',
        'seq_len': 25,
        'stride': 5,    
    }
}

MMNIST_10_10 = {
    'meta': {
        'dataset': 'Moving-MNIST',
        'seq_len': 10,
        'out_len': 10,
        'metrics': ['vpmae', 'vpmse', 'ssim', 'psnr', 'lpips', 'fss', 'rhd']
    },
    'param': {            
        'data_root': 'data',
    }
}


SMMNIST_10_10 = {
    'meta': {
        'dataset': 'SMoving-MNIST',
        'seq_len': 10,
        'out_len': 10,
        'metrics': ['vpmae', 'vpmse', 'ssim', 'psnr', 'lpips', 'fss', 'rhd']
    },
    'param': {            
        'data_root': 'data',
    }
}

METEO_4_12 = {
    'meta': {
        'dataset': 'meteo',
        'seq_len': 4,
        'out_len': 12,
        'reshape': 256,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips', 'csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219', 
                    'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219', 
                    'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219', 'fss', 'rhd'],
    },
    'param': {
        'in_len': 4,
        'stride': 4,
        'out_len': 12,
    },
    'savedir': 'meteo'
}

METEO_12_12 = {
    'meta': {
        'dataset': 'meteo',
        'seq_len': 12,
        'out_len': 12,
        'reshape': 256,
        'metrics': ['mae', 'mse', 'ssim', 'psnr', 'lpips', 'csi-16', 'csi-74', 'csi-133', 'csi-160', 'csi-181', 'csi-219', 
                    'csi_4-16', 'csi_4-74', 'csi_4-133', 'csi_4-160', 'csi_4-181', 'csi_4-219', 
                    'csi_16-16', 'csi_16-74', 'csi_16-133', 'csi_16-160', 'csi_16-181', 'csi_16-219', 'fss', 'rhd'],
    },
    'param': {
        'in_len': 12,
        'stride': 12,
        'out_len': 12,
    },
    'savedir': 'meteo'
}