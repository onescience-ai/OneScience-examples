import numpy as np
import xarray as xr
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn.init import trunc_normal_






def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNOBlock3d(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            data_format="channels_last",
            mlp_out_features=None,
        ):
        super().__init__()
        self.norm_layer = norm_layer
        self.norm1 = norm_layer(dim)
        self.filter = AFNO3D(dim, num_blocks, sparsity_threshold, 
            hard_thresholding_fraction) 
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, out_features=mlp_out_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        self.double_skip = double_skip
        #self.channels_first = (data_format == "channels_first")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        #if self.channels_first:
            # AFNO natively uses a channels-last data format 
        #x = x.permute(0,1,3,4,2)
        x = x.permute(0,2,3,4,1)

        residual = x

        x = self.norm1(x)

        x = self.filter(x)


        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)

        x = self.mlp(x)

        x = x + residual

        # if self.channels_first:
        #x = x.permute(0,1,4,2,3)
        x = x.permute(0,4,1,2,3)

        return x


class AFNO3D(nn.Module):
    def __init__(
        self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
        hard_thresholding_fraction=1, hidden_size_factor=1
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, D, H, W, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, D, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, D, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)


        # Calculate intermediate values out-of-place
        # Clone the full tensors to work out-of-place
        o1_real_clone = o1_real.clone()
        o1_imag_clone = o1_imag.clone()
        o2_real_clone = o2_real.clone()
        o2_imag_clone = o2_imag.clone()

        o1_real_out = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + 
            self.b1[0]
        )

        o1_imag_out = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + 
            torch.einsum('...bi,bio->...bo', x[:, :, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real_out = (
            torch.einsum('...bi,bio->...bo', o1_real_out, self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag_out, self.w2[1]) + 
            self.b2[0]
        )

        o2_imag_out = (
            torch.einsum('...bi,bio->...bo', o1_imag_out, self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real_out, self.w2[1]) + 
            self.b2[1]
        )

        # Define the starting and ending slices for the updates
        start, end = total_modes - kept_modes, total_modes + kept_modes

        # Separate the unmodified parts of `o1_real_clone` and insert `o1_real_out` into the middle
        o1_real_new = torch.cat([
            o1_real_clone[:, :, :start, :],                # Unmodified beginning part
            o1_real_out,                                   # Updated middle part
            o1_real_clone[:, :, end:, :]                   # Unmodified ending part
        ], dim=2)

       # Repeat the same logic for o1_imag, o2_real, and o2_imag
        o1_imag_new = torch.cat([
            o1_imag_clone[:, :, :start, :],
            o1_imag_out,
            o1_imag_clone[:, :, end:, :]
        ], dim=2)

        o2_real_new = torch.cat([
            o2_real_clone[:, :, :start, :],
            o2_real_out,
            o2_real_clone[:, :, end:, :]
        ], dim=2)

        o2_imag_new = torch.cat([
            o2_imag_clone[:, :, :start, :],
            o2_imag_out,
            o2_imag_clone[:, :, end:, :]
        ], dim=2)

        # Reassign the new tensors back to the original variables
        o1_real, o1_imag, o2_real, o2_imag = o1_real_new, o1_imag_new, o2_real_new, o2_imag_new

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, D, H, W // 2 + 1, C)
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(1,2,3), norm="ortho")
        x = x.type(dtype)

        return x + bias
    





class Interpolate(nn.Module):
    def __init__(self, scale, mode, corners=None, antialias=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.align_corners = corners
        self.antialias = antialias

    def forward(self, x):

        if (self.mode == 'bicubic' or self.mode == 'bilinear'):
            B, C, T, W, H = x.size() 
            x = x.reshape(B, C*T, W, H)
            x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners, antialias=self.antialias)
            #x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners, antialias=self.antialias)
            x = x.reshape(B, C, T, int(W*self.scale[0]), int(H*self.scale[1]))

        else:
            x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners, antialias=self.antialias)
            #x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners, antialias=self.antialias)
        return x




class AFNOCast(nn.Module):
    def __init__(self, n_feature=16, dropout_state=1, n_ens=51, bias=2.0):
        '''
        n_feature: int
                number of features
        dropout_state: int
                0: no dropout, for random ensemble generation, 1: late_dropout, 2: early dropout
        '''
        super(AFNOCast, self).__init__()
        
        self.n_feature = n_feature
        self.dropout_state = dropout_state
        self.initial_dim = 1 #10
        self.n_ens = n_ens
        self.bias_value = bias
        
        self.apply(self._init_weights)

        #self.droupout = PermaDropout(0.1)
        self.dropout = nn.Dropout3d(p=0.1)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_feature, 11, 106, 78))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.afno = AFNOBlock3d(dim=self.n_feature, 
                         mlp_ratio=4., 
                         drop=0, 
                         drop_path=0, 
                         norm_layer=norm_layer,
                         num_blocks=2, 
                         sparsity_threshold=0.001, 
                         hard_thresholding_fraction=1.0
                        ) 
        #self.perma_dropout = PermaDropout(rate=0.1)


      # encoder
        encoder_hidden_dim = self.n_feature
        current_dim = self.initial_dim
        encoder_modules = []
        for i in range(2):
            padding = 1  # Reflection padding
            encoder_modules.append(nn.ReflectionPad3d(padding))
            encoder_modules.append(nn.Conv3d(current_dim, encoder_hidden_dim, 3, bias=True))
            encoder_modules.append(nn.ReLU())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv3d(current_dim, self.n_feature, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)
        
        # encoder2

        encoder2_modules = []
        for i in range(2):
            padding = 1  # Reflection padding
            encoder2_modules.append(nn.ReflectionPad3d([padding, padding, padding, padding, 1, 1]))
            encoder2_modules.append(nn.Conv3d(self.n_feature, self.n_feature, (3,3,3), bias=True))##, groups=self.n_feature))
            encoder2_modules.append(nn.ReLU())
        encoder2_modules.append(nn.Conv3d(self.n_feature, self.n_feature, 1, bias=False))#, groups=self.n_feature))
        self.encoder2 = nn.Sequential(*encoder2_modules)


        # decoder
        decoder_hidden_dim = self.n_feature
        current_dim = self.n_feature # + self.big_skip*self.in_chans
        decoder_modules = []
        for i in range(2):
            padding = 1  # Reflection padding
            decoder_modules.append(nn.ReflectionPad3d(padding))
            decoder_modules.append(nn.Conv3d(current_dim, decoder_hidden_dim, 3, bias=True, stride=(4,1,1)))#, groups=self.n_feature))
            decoder_modules.append(nn.ReLU())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv3d(current_dim, self.n_ens, 1, bias=False))#, groups=self.n_feature))
        self.decoder = nn.Sequential(*decoder_modules)   
        
        self.learnable_bias = nn.Parameter(torch.full([ 1, 1, 106, 78], self.bias_value))#, device='cuda'))
        
        self.ensemble_std_global = None  # Initialize global variable for ensemble std as None
        
        self.final = nn.ReLU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            #nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def process_chain(self, x):
        
        std_dev = self.ensemble_std_global 
        std_dev = torch.unsqueeze(std_dev, 1)
        

        # Processing for a single channel
        x = torch.unsqueeze(x, 1)

        residual_0 = x

        x = self.encoder(x)
        residual_1 = x
        x = self.encoder2(x)


        # encode cond
        std_dev = self.encoder(std_dev)

        # add/concat cond (std_dev)
        x = x + residual_1 + std_dev
        
        if self.dropout_state == 2:
            x = self.dropout(x)
        
        # Interpolating
        x = Interpolate(scale=(2.79, 2.79), mode='bicubic', corners=False, antialias=True)(x)    
        residual_0 = Interpolate(scale=(2.79, 2.79), mode='bicubic', corners=False, antialias=True)(residual_0)
        residual_1 = Interpolate(scale=(2.79, 2.79), mode='bicubic', corners=False, antialias=True)(residual_1)
        
        
        # Positional embedding and afno layer
        x = x + self.pos_embed
        # add noise to pos embed
        # noise_level = 0.3
        # x = x + torch.randn_like(x) * noise_level

        
        x = self.afno(x)
        
        if self.dropout_state == 1:
            x = self.dropout(x)
        
        # Final layers
        x = x + residual_0
        x = self.encoder2(x)
        #x = x + self.learnable_bias
        x = x + residual_1
        x = self.decoder(x)
        x = torch.squeeze(x, 1)
        
        #x = x + self.learnable_bias
        
        
        return self.final(x)
    

   

    def forward(self, x_in, cond):
        
        self.ensemble_std_global = cond#.to(device)  # Store std dev globally
   
        # # Apply process_chain across the ensemble dimension (dim=1)
        output = torch.vmap(self.process_chain, in_dims=1, out_dims=1, randomness="different")(x_in) 
    
        return output
    
    ######### Training - Prediction - Testing functions ###########

    def predict_step(self, x, x2, device):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            x2 = x2.to(device)
            # Calc prediction
            preds = self(x, x2)
        return preds
    
    def test_step(self, x, x2, y, device, loss_config={"crps": {"weight": 1.0}}):
        from seasonal_afnocast.train.losses import compute_losses
        self.eval()
        with torch.no_grad():
            # Prepare data
            x = x.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            # Calc prediction
            pred = self(x, x2)

            # Compute losses using unified system
            losses = compute_losses(pred, y, loss_config)
            # Convert to scalars for logging
            losses_scalar = {k: v.item() for k, v in losses.items()}
        
        return losses_scalar, pred, y
    
    def train_step (self, x, x2, y, device, optimizer, loss_config = {"crps": {"weight": 1.0}}):
        from seasonal_afnocast.train.losses import compute_losses
        """
        Runs a single training step with unified loss computation.
        Args:
            x (Tensor): Input tensor of shape (B, 1, 10, 47, 121, 240).
            y (Tensor): Target tensor of shape (B, 2, 121, 240).
            device (torch.device)
            optimizer
            loss_config (dict): Loss configuration with weights, e.g. {"mse": {"weight": 1.0}}

        Returns: dict of losses, predictions, targets
        """
               
        self.train()

        # Clear gradients
        optimizer.zero_grad()


        # Prepare data
        x = x.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Calc prediction
        pred = self(x, x2)

        # Compute losses using unifies system
        losses = compute_losses(pred, y, loss_config)

        # losses = S2S_loss.crps(pred, data_labels) # To Do: Check if my CRPS loss implementation is different


        # Backprop on the weighted total loss
        losses["total"].backward()
                        
        #scaler.scale(losses["crps"]).backward() # in case of Automated Mixed Precision

        # Include clipping of gradients to prevent exploding of gradients producing Nans
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)



        # in case of AMP
        # scaler.step(optimizer)
        # scaler.update()

        # dealing with memory allocation issues
        torch.cuda.empty_cache()

        # Conver to scalers for logging
        losses_scalar = {k: v.item() for k, v in losses.items()}

        return losses_scalar, pred, y
    
    
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.g + self.b

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        padding = 1  # Reflection padding
        self.reflection = nn.ReflectionPad3d(padding)
        self.proj = nn.Conv3d(dim, dim_out, 3, bias=True)
        #self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = ChanLayerNorm(dim_out)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift = None):
        x = self.reflection(x)
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        cond_dim = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.mlp = None

        if exists(cond_dim):
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond = None):

        scale_shift = None

        assert not (exists(self.mlp) ^ exists(cond))

        if exists(self.mlp) and exists(cond):
            cond = self.mlp(cond)
            cond = rearrange(cond, 'b c -> b c 1 1')
            scale_shift = cond.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class ResnetBlocks(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_in = None,
        depth = 1,
        cond_dim = None
    ):
        super().__init__()
        curr_dim = default(dim_in, dim)

        blocks = []
        for _ in range(depth):
            blocks.append(ResnetBlock(dim = curr_dim, dim_out = dim, cond_dim = cond_dim))
            curr_dim = dim

        self.blocks = ModuleList(blocks)

    def forward(self, x, cond = None):

        for block in self.blocks:
            x = block(x, cond = cond)

        return x
    
# From PyTorch internals
# from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/helpers.py
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
    
class PatchEmbed(nn.Module):
    """
    from https://github.com/NVlabs/AFNO-transformer/blob/master/classification/afnonet.py
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim=768):
        super().__init__()
        #to_2tuple = _ntuple(2)
        #img_size = self.to_2tuple(img_size)
        #patch_size = self.to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


   
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    
        
        
# Example usage
if __name__ == "__main__":
    # Example usage
    model = AFNOCast(n_feature=10, dropout_state=1, n_ens=1)
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)
    model.train()

    model.to(device)
    # Print model summary
    # summary(model,  input_size=[ (1,25, 11, 38, 28), (1,11,38,28)])
    #summary(model, input_data = [torch.randn(20 ,10, 11, 38, 28).to(device), torch.randint(0,365, (20,))])

    # Test script

    # Create input tensor with unique constants for each ensemble member
    batch_size, ensemble_size, time, height, width = 1, 25, 11, 38, 28
    x_in = torch.zeros(batch_size, ensemble_size, time, height, width).to(device)
    for i in range(ensemble_size):
        x_in[:, i, :, :, :] = i  # Fill with 0, 1, 2, ..., 24
    cond = torch.zeros(1, time, height, width).to(device)  # Zero condition

    # Check input variability
    print("Input x_in shape:", x_in.shape)
    print("Input means across ensemble dim:", x_in.mean(dim=(0, 2, 3, 4)).cpu().numpy())

    # Run forward pass
    print("\nRunning vmap, checking input to process_chain:")
    output = model(x_in, cond)
    print("\nOutput shape:", output.shape)
    print("Output ensemble std:", torch.std(output, dim=1).mean().item())

    # Manual loop for comparison
    manual_output = []
    for i in range(ensemble_size):
        single_input = x_in[:, i, :, :, :].to(device)
        single_output = model.process_chain(single_input)
        manual_output.append(single_output)
    manual_output = torch.stack(manual_output, dim=1)
    print("Manual output shape:", manual_output.shape)
    print("Manual output ensemble std:", torch.std(manual_output, dim=1).mean().item())
    print("Outputs match:", torch.allclose(output, manual_output, atol=1e-5))
        

