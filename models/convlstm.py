'''
https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
'''

import torch
import torch.nn as nn

from collections import OrderedDict

class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device='cuda'):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=5):
        if inputs is not None:
            assert seq_len == inputs.size(0), f'forward seq_len: {seq_len}, actual: {inputs.size(0)}'

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height, self._state_width), dtype=torch.float).to(inputs.get_device())
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height, self._state_width), dtype=torch.float).to(inputs.get_device())
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height, self._state_width), dtype=torch.float).to(c.get_device())
            else:
                x = inputs[index, ...]
            # print(x.shape, h.shape)
            cat_x = torch.cat([x, h], dim=1)            
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)
    
def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, None, seq_len=seq_number)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))            
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, out_len=20):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.out_len = out_len
        self.blocks = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=self.out_len)        
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)), getattr(self, 'rnn' + str(i)))
        return input

class EncoderForecaster(nn.Module):

    def __init__(self, encoder, forecaster, last_activation='none'):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.last_activation = last_activation

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        if self.last_activation == 'sigmoid':
            output = torch.sigmoid(output)
        return output
