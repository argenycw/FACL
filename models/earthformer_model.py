import torch
from bisect import bisect_right

from omegaconf import OmegaConf
from earthformer.cuboid_transformer.cuboid_transformer import *

class SequentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self.optimizer = optimizer
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda

class CuboidTransformerDecoder(nn.Module):
    """Decoder of the CuboidTransformer.

    For each block, we first apply the StackCuboidSelfAttention and then apply the StackCuboidCrossAttention

    Repeat the following structure K times

        x --> StackCuboidSelfAttention --> |
                                           |----> StackCuboidCrossAttention (If used) --> out
                                   mem --> |

    """
    def __init__(self,
                 target_temporal_length,
                 mem_shapes,
                 cross_start=0,
                 depth=[2, 2],
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 block_self_attn_patterns=None,
                 block_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 block_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 block_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 block_cross_attn_patterns=None,
                 block_cross_cuboid_hw=[(4, 4), (4, 4)],
                 block_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 block_cross_shift_hw=[(0, 0), (0, 0)],
                 block_cross_n_temporal=[1, 2],
                 cross_last_n_frames=None,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=False,
                 hierarchical_pos_embed=False,
                 pos_embed_type='t+hw',
                 max_temporal_relative=50,
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 use_first_self_attn=False,
                 # global vectors
                 use_self_global=False,
                 self_update_global=True,
                 use_cross_global=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 up_linear_init_mode="0",
                 norm_init_mode="0",
                 **kwargs):
        """

        Parameters
        ----------
        target_temporal_length
        mem_shapes
        cross_start
            The block to start cross attention
        depth
            Depth of each block
        upsample_type
            The type of the upsampling layers
        upsample_kernel_size
        block_self_attn_patterns
            Pattern of the block self attentions
        block_self_cuboid_size
        block_self_cuboid_strategy
        block_self_shift_size
        block_cross_attn_patterns
        block_cross_cuboid_hw
        block_cross_cuboid_strategy
        block_cross_shift_hw
        block_cross_n_temporal
        num_heads
        attn_drop
        proj_drop
        ffn_drop
        ffn_activation
        gated_ffn
        norm_layer
        use_inter_ffn
        hierarchical_pos_embed
            Whether to add pos embedding for each hierarchy.
        max_temporal_relative
        padding_type
        checkpoint_level
        """
        super(CuboidTransformerDecoder, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.up_linear_init_mode = up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(depth) == len(mem_shapes)
        self.target_temporal_length = target_temporal_length
        self.num_blocks = len(mem_shapes)
        self.cross_start = cross_start
        self.mem_shapes = mem_shapes
        self.depth = depth
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_first_self_attn = use_first_self_attn
        if block_self_attn_patterns is not None:
            if isinstance(block_self_attn_patterns, (tuple, list)):
                assert len(block_self_attn_patterns) == self.num_blocks
            else:
                block_self_attn_patterns = [block_self_attn_patterns for _ in range(self.num_blocks)]
            block_self_cuboid_size = []
            block_self_cuboid_strategy = []
            block_self_shift_size = []
            for idx, key in enumerate(block_self_attn_patterns):
                func = CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_self_cuboid_size.append(cuboid_size)
                block_self_cuboid_strategy.append(strategy)
                block_self_shift_size.append(shift_size)
        else:
            if not isinstance(block_self_cuboid_size[0][0], (list, tuple)):
                block_self_cuboid_size = [block_self_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_size={block_self_cuboid_size}'

            if not isinstance(block_self_cuboid_strategy[0][0], (list, tuple)):
                block_self_cuboid_strategy = [block_self_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_cuboid_strategy={block_self_cuboid_strategy}'

            if not isinstance(block_self_shift_size[0][0], (list, tuple)):
                block_self_shift_size = [block_self_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_self_shift_size={block_self_shift_size}'
        self_blocks = []
        for i in range(self.num_blocks):
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use an additional self attention layer.
                ele_depth = depth[i] - 1
            else:
                ele_depth = depth[i]
            stack_cuboid_blocks =\
                [StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_self_cuboid_size[i],
                    block_strategy=block_self_cuboid_strategy[i],
                    block_shift_size=block_self_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_self_global,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(ele_depth)]
            self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.self_blocks = nn.ModuleList(self_blocks)

        if block_cross_attn_patterns is not None:
            if isinstance(block_cross_attn_patterns, (tuple, list)):
                assert len(block_cross_attn_patterns) == self.num_blocks
            else:
                block_cross_attn_patterns = [block_cross_attn_patterns for _ in range(self.num_blocks)]

            block_cross_cuboid_hw = []
            block_cross_cuboid_strategy = []
            block_cross_shift_hw = []
            block_cross_n_temporal = []
            for idx, key in enumerate(block_cross_attn_patterns):
                if key == "last_frame_dst":
                    cuboid_hw = None
                    shift_hw = None
                    strategy = None
                    n_temporal = None
                else:
                    func = CuboidCrossAttentionPatterns.get(key)
                    cuboid_hw, shift_hw, strategy, n_temporal = func(mem_shapes[idx])
                block_cross_cuboid_hw.append(cuboid_hw)
                block_cross_cuboid_strategy.append(strategy)
                block_cross_shift_hw.append(shift_hw)
                block_cross_n_temporal.append(n_temporal)
        else:
            if not isinstance(block_cross_cuboid_hw[0][0], (list, tuple)):
                block_cross_cuboid_hw = [block_cross_cuboid_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_hw={block_cross_cuboid_hw}'

            if not isinstance(block_cross_cuboid_strategy[0][0], (list, tuple)):
                block_cross_cuboid_strategy = [block_cross_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_strategy) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_cuboid_strategy={block_cross_cuboid_strategy}'

            if not isinstance(block_cross_shift_hw[0][0], (list, tuple)):
                block_cross_shift_hw = [block_cross_shift_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_shift_hw) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_shift_hw={block_cross_shift_hw}'
            if not isinstance(block_cross_n_temporal[0], (list, tuple)):
                block_cross_n_temporal = [block_cross_n_temporal for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_n_temporal) == self.num_blocks, \
                    f'Incorrect input format! Received block_cross_n_temporal={block_cross_n_temporal}'
        self.cross_blocks = nn.ModuleList()
        for i in range(self.cross_start, self.num_blocks):
            cross_block = nn.ModuleList(
                [StackCuboidCrossAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_hw=block_cross_cuboid_hw[i],
                    block_strategy=block_cross_cuboid_strategy[i],
                    block_shift_hw=block_cross_shift_hw[i],
                    block_n_temporal=block_cross_n_temporal[i],
                    cross_last_n_frames=cross_last_n_frames,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    activation=ffn_activation,
                    max_temporal_relative=max_temporal_relative,
                    padding_type=padding_type,
                    use_global_vector=use_cross_global,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(depth[i])])
            self.cross_blocks.append(cross_block)

        # Construct upsampling layers
        if self.num_blocks > 1:
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.ModuleList([
                    Upsample3DLayer(
                        dim=self.mem_shapes[i + 1][-1],
                        out_dim=self.mem_shapes[i][-1],
                        target_size=(target_temporal_length,) + self.mem_shapes[i][1:3],
                        kernel_size=upsample_kernel_size,
                        temporal_upsample=False,
                        conv_init_mode=conv_init_mode,
                    )
                    for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.hierarchical_pos_embed:
                self.hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.mem_shapes[i][-1], typ=pos_embed_type,
                             maxT=target_temporal_length, maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])

        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.cross_blocks:
            for m in ms:
                m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        """

        Parameters
        ----------
        x
            Shape (B, T_top, H_top, W_top, C)
        mem_l
            A list of memory tensors

        Returns
        -------
        out
        """
        B, T_top, H_top, W_top, C = x.shape
        assert T_top == self.target_temporal_length
        assert (H_top, W_top) == (self.mem_shapes[-1][1], self.mem_shapes[-1][2])
        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = None if mem_global_vector_l is None else mem_global_vector_l[i]
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # For the top block, we won't use the self attention layer and will directly use the cross attention layer.
                if i >= self.cross_start:
                    x = self.cross_blocks[i - self.cross_start][0](x, mem_l[i], mem_global_vector)
                for idx in range(self.depth[i] - 1):
                    if self.use_self_global:  # in this case `mem_global_vector` is guaranteed to be not None
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx + 1](x, mem_l[i], mem_global_vector)
            else:
                for idx in range(self.depth[i]):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
            # Upsample
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.hierarchical_pos_embed_l[i - 1](x)
        return x

class CuboidTransformerModified(nn.Module):
    """Cuboid Transformer for spatiotemporal forecasting

    We adopt the Non-autoregressive encoder-decoder architecture.
    The decoder takes the multi-scale memory output from the encoder.

    The initial downsampling / upsampling layers will be
    Downsampling: [K x Conv2D --> PatchMerge]
    Upsampling: [Nearest Interpolation-based Upsample --> K x Conv2D]

    x --> downsample (optional) ---> (+pos_embed) ---> enc --> mem_l         initial_z (+pos_embed) ---> FC
                                                     |            |
                                                     |------------|
                                                           |
                                                           |
             y <--- upsample (optional) <--- dec <----------

    """
    def __init__(self,
                 input_shape,
                 target_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 # inter-attn downsample/upsample
                 downsample=2,
                 downsample_type='patch_merge',
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 # encoder
                 enc_depth=[4, 4, 4],
                 enc_attn_patterns=None,
                 enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 enc_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 enc_shift_size=[(0, 0, 0), (0, 0, 0)],
                 enc_use_inter_ffn=True,
                 # decoder
                 dec_depth=[2, 2],
                 dec_cross_start=0,
                 dec_self_attn_patterns=None,
                 dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 dec_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
                 dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
                 dec_cross_attn_patterns=None,
                 dec_cross_cuboid_hw=[(4, 4), (4, 4)],
                 dec_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
                 dec_cross_shift_hw=[(0, 0), (0, 0)],
                 dec_cross_n_temporal=[1, 2],
                 dec_cross_last_n_frames=None,
                 dec_use_inter_ffn=True,
                 dec_hierarchical_pos_embed=False,
                 # global vectors
                 num_global_vectors=4,
                 use_dec_self_global=True,
                 dec_self_update_global=True,
                 use_dec_cross_global=True,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 z_init_method='nearest_interp',
                 # # initial downsample and final upsample
                 initial_downsample_type="conv",
                 initial_downsample_activation="leaky",
                 # initial_downsample_type=="conv"
                 initial_downsample_scale=1,
                 initial_downsample_conv_layers=2,
                 final_upsample_conv_layers=2,
                 # initial_downsample_type == "stack_conv"
                 initial_downsample_stack_conv_num_layers=1,
                 initial_downsample_stack_conv_dim_list=None,
                 initial_downsample_stack_conv_downscale_list=[1, ],
                 initial_downsample_stack_conv_num_conv_list=[2, ],
                 # # end of initial downsample and final upsample
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 padding_type='ignore',
                 pos_embed_type='t+hw',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 dec_use_first_self_attn=False,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 conv_init_mode="0",
                 down_up_linear_init_mode="0",
                 norm_init_mode="0",
                 **kwargs, # TODO this is added additional to avoid errors
                 ):
        """

        Parameters
        ----------
        input_shape
            Shape of the input tensor. It will be (T, H, W, C_in)
        target_shape
            Shape of the input tensor. It will be (T_out, H, W, C_out)
        base_units
            The base units
        z_init_method
            How the initial input to the decoder is initialized
        """
        super(CuboidTransformerModified, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_up_linear_init_mode = down_up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(enc_depth) == len(dec_depth)
        self.base_units = base_units
        self.num_global_vectors = num_global_vectors
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.z_init_method = z_init_method
        assert self.z_init_method in ['zeros', 'nearest_interp', 'last', 'mean']

        self.input_shape = input_shape
        self.target_shape = target_shape
        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out

        if self.num_global_vectors > 0:
            self.init_global_vectors = nn.Parameter(
                torch.zeros((self.num_global_vectors, global_dim_ratio*base_units)))

        new_input_shape = self.get_initial_encoder_final_decoder(
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_type=initial_downsample_type,
            activation=initial_downsample_activation,
            # initial_downsample_type=="conv"
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            padding_type=padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers=initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list=initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list=initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list=initial_downsample_stack_conv_num_conv_list,
        )
        T_in, H_in, W_in, _ = new_input_shape

        self.encoder = CuboidTransformerEncoder(
            input_shape=(T_in, H_in, W_in, base_units),
            base_units=base_units,
            block_units=block_units,
            scale_alpha=scale_alpha,
            depth=enc_depth,
            downsample=downsample,
            downsample_type=downsample_type,
            block_attn_patterns=enc_attn_patterns,
            block_cuboid_size=enc_cuboid_size,
            block_strategy=enc_cuboid_strategy,
            block_shift_size=enc_shift_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            gated_ffn=gated_ffn,
            ffn_activation=ffn_activation,
            norm_layer=norm_layer,
            use_inter_ffn=enc_use_inter_ffn,
            padding_type=padding_type,
            use_global_vector=num_global_vectors > 0,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.enc_pos_embed = PosEmbed(
            embed_dim=base_units, typ=pos_embed_type,
            maxH=H_in, maxW=W_in, maxT=T_in)
        mem_shapes = self.encoder.get_mem_shapes()

        self.z_proj = nn.Linear(mem_shapes[-1][-1], mem_shapes[-1][-1])
        self.dec_pos_embed = PosEmbed(
            embed_dim=mem_shapes[-1][-1], typ=pos_embed_type,
            maxT=T_out, maxH=mem_shapes[-1][1], maxW=mem_shapes[-1][2])
        self.decoder = CuboidTransformerDecoder(
            target_temporal_length=T_out,
            mem_shapes=mem_shapes,
            cross_start=dec_cross_start,
            depth=dec_depth,
            upsample_type=upsample_type,
            block_self_attn_patterns=dec_self_attn_patterns,
            block_self_cuboid_size=dec_self_cuboid_size,
            block_self_shift_size=dec_self_shift_size,
            block_self_cuboid_strategy=dec_self_cuboid_strategy,
            block_cross_attn_patterns=dec_cross_attn_patterns,
            block_cross_cuboid_hw=dec_cross_cuboid_hw,
            block_cross_shift_hw=dec_cross_shift_hw,
            block_cross_cuboid_strategy=dec_cross_cuboid_strategy,
            block_cross_n_temporal=dec_cross_n_temporal,
            cross_last_n_frames=dec_cross_last_n_frames,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            upsample_kernel_size=upsample_kernel_size,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            use_inter_ffn=dec_use_inter_ffn,
            max_temporal_relative=T_in + T_out,
            padding_type=padding_type,
            hierarchical_pos_embed=dec_hierarchical_pos_embed,
            pos_embed_type=pos_embed_type,
            use_self_global=(num_global_vectors > 0) and use_dec_self_global,
            self_update_global=dec_self_update_global,
            use_cross_global=(num_global_vectors > 0) and use_dec_cross_global,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            use_first_self_attn=dec_use_first_self_attn,
            # initialization
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
            #**kwargs, # TODO this is added additional to avoid errors
        )
        self.reset_parameters()

    def get_initial_encoder_final_decoder(
            self,
            initial_downsample_type,
            activation,
            # initial_downsample_type=="conv"
            initial_downsample_scale,
            initial_downsample_conv_layers,
            final_upsample_conv_layers,
            padding_type,
            # initial_downsample_type == "stack_conv"
            initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list,
        ):
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        # Construct the initial upsampling / downsampling layers
        self.initial_downsample_type = initial_downsample_type
        if self.initial_downsample_type == "conv":
            if isinstance(initial_downsample_scale, int):
                initial_downsample_scale = (1, initial_downsample_scale, initial_downsample_scale)
            elif len(initial_downsample_scale) == 2:
                initial_downsample_scale = (1, *initial_downsample_scale)
            elif len(initial_downsample_scale) == 3:
                initial_downsample_scale = tuple(initial_downsample_scale)
            else:
                raise NotImplementedError(f"initial_downsample_scale {initial_downsample_scale} format not supported!")
            # if any(ele > 1 for ele in initial_downsample_scale):
            self.initial_encoder = InitialEncoder(dim=C_in,
                                                  out_dim=self.base_units,
                                                  downsample_scale=initial_downsample_scale,
                                                  num_conv_layers=initial_downsample_conv_layers,
                                                  padding_type=padding_type,
                                                  activation=activation,
                                                  conv_init_mode=self.conv_init_mode,
                                                  linear_init_mode=self.down_up_linear_init_mode,
                                                  norm_init_mode=self.norm_init_mode)
            self.final_decoder = FinalDecoder(dim=self.base_units,
                                              target_thw=(T_out, H_out, W_out),
                                              num_conv_layers=final_upsample_conv_layers,
                                              activation=activation,
                                              conv_init_mode=self.conv_init_mode,
                                              linear_init_mode=self.down_up_linear_init_mode,
                                              norm_init_mode=self.norm_init_mode)
            new_input_shape = self.initial_encoder.patch_merge.get_out_shape(self.input_shape)
            self.dec_final_proj = nn.Linear(self.base_units, C_out)
            # else:
            #     self.initial_encoder = nn.Linear(C_in, self.base_units)
            #     self.final_decoder = nn.Identity()
            #     self.dec_final_proj = nn.Linear(self.base_units, C_out)
            #     new_input_shape = self.input_shape

        elif self.initial_downsample_type == "stack_conv":
            if initial_downsample_stack_conv_dim_list is None:
                initial_downsample_stack_conv_dim_list = [self.base_units, ] * initial_downsample_stack_conv_num_layers
            self.initial_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=C_in,
                out_dim_list=initial_downsample_stack_conv_dim_list,
                downsample_scale_list=initial_downsample_stack_conv_downscale_list,
                num_conv_per_merge_list=initial_downsample_stack_conv_num_conv_list,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            # use `self.target_shape` to get correct T_out
            initial_encoder_out_shape_list = self.initial_encoder.get_out_shape_list(self.target_shape)
            dec_target_shape_list, dec_in_dim = \
                FinalStackUpsamplingDecoder.get_init_params(
                    enc_input_shape=self.target_shape,
                    enc_out_shape_list=initial_encoder_out_shape_list,
                    large_channel=True)
            self.final_decoder = FinalStackUpsamplingDecoder(
                target_shape_list=dec_target_shape_list,
                in_dim=dec_in_dim,
                num_conv_per_up_list=initial_downsample_stack_conv_num_conv_list[::-1],
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode)
            self.dec_final_proj = nn.Linear(dec_target_shape_list[-1][-1], C_out)
            new_input_shape = self.initial_encoder.get_out_shape_list(self.input_shape)[-1]
        else:
            raise NotImplementedError
        self.input_shape_after_initial_downsample = new_input_shape
        T_in, H_in, W_in, _ = new_input_shape

        return new_input_shape

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            nn.init.trunc_normal_(self.init_global_vectors, std=.02)
        if hasattr(self.initial_encoder, "reset_parameters"):
            self.initial_encoder.reset_parameters()
        else:
            apply_initialization(self.initial_encoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        if hasattr(self.final_decoder, "reset_parameters"):
            self.final_decoder.reset_parameters()
        else:
            apply_initialization(self.final_decoder,
                                 conv_mode=self.conv_init_mode,
                                 linear_mode=self.down_up_linear_init_mode,
                                 norm_mode=self.norm_init_mode)
        apply_initialization(self.dec_final_proj,
                             linear_mode=self.down_up_linear_init_mode)
        self.encoder.reset_parameters()
        self.enc_pos_embed.reset_parameters()
        self.decoder.reset_parameters()
        self.dec_pos_embed.reset_parameters()
        apply_initialization(self.z_proj,
                             linear_mode="0")

    def get_initial_z(self, final_mem, T_out):
        B = final_mem.shape[0]
        if self.z_init_method == 'zeros':
            z_shape = (1, T_out) + final_mem.shape[2:]
            initial_z = torch.zeros(z_shape, dtype=final_mem.dtype, device=final_mem.device)
            initial_z = self.z_proj(self.dec_pos_embed(initial_z)).expand(B, -1, -1, -1, -1)
        elif self.z_init_method == 'nearest_interp':
            # final_mem will have shape (B, T, H, W, C)
            initial_z = F.interpolate(final_mem.permute(0, 4, 1, 2, 3),
                                      size=(T_out, final_mem.shape[2], final_mem.shape[3])).permute(0, 2, 3, 4, 1)
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'last':
            initial_z = torch.broadcast_to(final_mem[:, -1:, :, :, :], (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == 'mean':
            initial_z = torch.broadcast_to(final_mem.mean(axis=1, keepdims=True),
                                           (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        else:
            raise NotImplementedError
        return initial_z

    def forward(self, x, verbose=False):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)
        verbos
            if True, print intermediate shapes
        Returns
        -------
        out
            The output Shape (B, T_out, H, W, C_out)
        """
        # print('Start forward in Cuboid Transformer...')
        B, _, _, _, _ = x.shape
        T_out = self.target_shape[0]
        x = self.initial_encoder(x)
        x = self.enc_pos_embed(x)
        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors\
                .expand(B, self.num_global_vectors, self.global_dim_ratio*self.base_units)
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
        else:
            mem_l = self.encoder(x)
        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        initial_z = self.get_initial_z(final_mem=mem_l[-1],
                                       T_out=T_out)
        if self.num_global_vectors > 0:
            dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(initial_z, mem_l)
        dec_out = self.final_decoder(dec_out)
        out = self.dec_final_proj(dec_out)
        return out



def get_model_config(input_shape=(5, 480, 480, 1), target_shape=(20, 480, 480, 1)):
    cfg = OmegaConf.create()
    # ====================================
    # model config
    # ====================================
    cfg.input_shape = input_shape
    cfg.target_shape = target_shape
    cfg.base_units = 64
    cfg.block_units = None # multiply by 2 when downsampling in each layer
    cfg.scale_alpha = 1.0

    cfg.num_heads = 4
    cfg.attn_drop = 0.1
    cfg.proj_drop = 0.1
    cfg.ffn_drop = 0.1    

    # inter-attn downsample/upsample
    cfg.downsample = 2
    cfg.downsample_type = "patch_merge"
    cfg.upsample_type = "upsample"
    # upsample_kernel_size ???    

    # encoder
    cfg.enc_depth = [1, 1]
    cfg.enc_attn_patterns = 'axial'
    # enc_cuboid_size ???
    # enc_cuboid_strategy ???
    # enc_shift_size ???    
    cfg.enc_use_inter_ffn = True

    # decoder
    cfg.dec_depth = [1, 1]
    # dec_cross_start ???
    cfg.dec_self_attn_patterns = 'axial'
    # dec_self_cuboid_size ???
    # dec_self_cuboid_strategy
    # dec_self_shift_size
    cfg.dec_cross_attn_patterns = 'cross_1x1'
    # dec_cross_cuboid_hw
    # dec_cross_cuboid_strategy
    # dec_cross_shift_hw
    # dec_cross_n_temporal
    cfg.dec_cross_last_n_frames = None 
    cfg.dec_use_inter_ffn = True
    cfg.dec_hierarchical_pos_embed = True    

    # global vectors
    cfg.num_global_vectors = 8
    cfg.use_dec_self_global = True
    cfg.dec_self_update_global = True
    cfg.use_dec_cross_global = True
    cfg.use_global_vector_ffn = True
    cfg.use_global_self_attn = False
    cfg.separate_global_qkv = False
    cfg.global_dim_ratio = 1
    cfg.z_init_method = 'zeros'
    # # initial downsample and final upsample
    cfg.initial_downsample_type = "stack_conv"
    cfg.initial_downsample_activation = "leaky"
    # initial_downsample_scale??
    # initial_downsample_conv_layers
    # final_upsample_conv_layers    
    cfg.initial_downsample_stack_conv_num_layers = 3
    cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
    cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
    cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
    # # end of initial downsample and final upsample
    cfg.ffn_activation = 'gelu'
    cfg.gated_ffn = False
    cfg.norm_layer = 'layer_norm'
    cfg.padding_type = 'zeros'
    cfg.pos_embed_type = "t+hw"
    cfg.checkpoint_level = 2
    cfg.use_relative_pos = True
    cfg.self_attn_use_final_proj = True
    cfg.dec_use_first_self_attn = False
    # initialization
    cfg.attn_linear_init_mode = "0"
    cfg.ffn_linear_init_mode = "0"
    cfg.conv_init_mode = "0"
    cfg.down_up_linear_init_mode = "0"
    cfg.norm_init_mode = "0"

    return cfg

def build_model(oc_file, input_shape=(5, 480, 480, 1), target_shape=(20, 480, 480, 1)):
    '''
    An overly simplistic implementation to build an Earthformer model.
    For the original version (700 lines+), please check train_cuboid_hko.py
    '''
    oc_from_file = OmegaConf.load(open(oc_file, "r"))
    oc = OmegaConf.create()
    oc.model = get_model_config(input_shape=input_shape, target_shape=target_shape)
    oc = OmegaConf.merge(oc, oc_from_file)
    model_cfg = OmegaConf.to_object(oc.model)

    model_cfg['enc_attn_patterns'] = [model_cfg["self_pattern"]] * 2
    model_cfg['dec_self_attn_patterns'] = [model_cfg["cross_self_pattern"]] * 2
    model_cfg['dec_cross_attn_patterns'] = [model_cfg["cross_pattern"]] * 2
    model_cfg['dec_hierarchical_pos_embed'] = False

    model = CuboidTransformerModified(**model_cfg)

    return model