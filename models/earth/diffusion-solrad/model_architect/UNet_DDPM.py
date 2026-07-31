import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock, AttnBlock
from .utils import get_named_beta_schedule

def sinusoidal_embedding(n, d):
    """
    n: iteration steps,
    d: time embedding dimension
    """
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

def _make_te(dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class UNet_with_time(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_frame = config.input_frame
        output_frame = config.output_frame
        n_steps = config.n_steps
        time_emb_dim = config.time_emb_dim
        cond_nc = config.cond_nc
        chs_mult = config.chs_mult ## e.g. (1, 2, 4, 8)
        n_res_blocks = config.n_res_blocks
        base_chs = config.base_chs
        ## e.g. (0, 0, 1, 1) -> 0 means no attention
        use_attn_list = config.use_attn_list 

        layer_depth = len(chs_mult)
        assert len(use_attn_list) == layer_depth, "length of use_attn_list should be the same as chs_mult"
        assert input_frame >= output_frame, "input_frame should be larger than or equal to output_frame"

        self.filter_list = [base_chs * m for m in chs_mult]

        ## time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.time_embed_fc = _make_te(time_emb_dim, time_emb_dim)
        ## end of time embedding

        ## input conv
        self.input_layer = nn.PixelUnshuffle(downscale_factor=2)

        ## downsampling
        self.down_blocks = nn.ModuleList()
        in_c = input_frame * 4 ## after pixel unshuffle
        for i in range(layer_depth):
            out_c = self.filter_list[i]

            for _ in range(n_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(in_c, in_c, cond_nc, time_emb_dim, down_flag=False, up_flag=False)
                )

            if use_attn_list[i]:
                self.down_blocks.append(AttnBlock(in_c, 4)) ## num_head=4
            
            self.down_blocks.append(
                ResidualBlock(in_c, out_c, cond_nc, time_emb_dim, down_flag=True, up_flag=False)
            )
            in_c = out_c
        ## end of downsampling

        ## middle
        self.mid_block1 = ResidualBlock(in_c, in_c, cond_nc, time_emb_dim, down_flag=False, up_flag=False)
        self.mid_attn = AttnBlock(in_c, 4)
        self.mid_block2 = ResidualBlock(in_c, in_c, cond_nc, time_emb_dim, down_flag=False, up_flag=False)
        ## end of middle

        ## upsampling
        self.up_blocks = nn.ModuleList()
        self.filter_list = [input_frame * 4] + self.filter_list[:-1]
        for i in reversed(range(layer_depth)): ## i = layer_depth-1, ..., 0
            out_c = self.filter_list[i]
            
            self.up_blocks.append(
                ResidualBlock(in_c*2, out_c, cond_nc, time_emb_dim, down_flag=False, up_flag=True)
            )
            if use_attn_list[i]:
                self.up_blocks.append(AttnBlock(out_c)) ## num_head=1

            for _ in range(n_res_blocks):
                self.up_blocks.append(
                    ResidualBlock(out_c*2, out_c, cond_nc, time_emb_dim, down_flag=False, up_flag=False)
                )

            in_c = out_c
        
        ## end of upsampling
        self.out_up = nn.PixelShuffle(upscale_factor=2)
        self.out_conv = nn.Conv2d(input_frame, output_frame, 3, padding=1)

    def forward(self, x, t, cond):
        """
        x: (b, in_c, h, w), noisy input (concatenated with some data)
        t: (b,), time step
        cond: (b, cond_nc, h, w), conditional input
        """
        # time embedding
        t_emb = self.time_embed(t) ## (b, time_emb_dim)
        t_emb = self.time_embed_fc(t_emb) ## (b, time_emb_dim)

        # input conv
        x = self.input_layer(x)

        # downsampling
        skip_x = []
        for ii, down_layer in enumerate(self.down_blocks):
            if isinstance(down_layer, ResidualBlock):
                x = down_layer(x, cond, t_emb)
                skip_x.append(x)
            elif isinstance(down_layer, AttnBlock):
                x = down_layer(x)
            else:
                raise ValueError("Wrong layer type in down_blocks")

        # middle
        x = self.mid_block1(x, cond, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond, t_emb)

        # upsampling
        for up_layer in self.up_blocks:
            if isinstance(up_layer, ResidualBlock):
                skip_feat = skip_x.pop()
                x = torch.cat([x, skip_feat], dim=1) ## concat along channel dimension
                x = up_layer(x, cond, t_emb)
            elif isinstance(up_layer, AttnBlock):
                x = up_layer(x)
            else:
                raise ValueError("Wrong layer type in up_blocks")

        # output
        x = self.out_up(x)
        x = self.out_conv(x)

        return x

class DDPM(nn.Module):
    def __init__(self, backbone, output_shape, n_steps=1000, min_beta=1e-4, max_beta=0.02, device='cuda'):
        """
        output_shape: dim(C, H, W)
        """
        super().__init__()
        self.device = device
        self.backbone_model = backbone
        self.output_shape  = output_shape

        self.n_steps = n_steps
        
        ## linear betas
        betas = get_named_beta_schedule("linear", n_steps, min_beta, max_beta)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def forward(self, x, t, cond):
        """
        x: (b, in_c, h, w), noisy input (concatenated with some data)
        cond: (b, cond_nc, h, w), conditional input
        t: (b,), time step
        """
        return self.backbone_model(x, t, cond)

    @torch.no_grad()
    def add_noise(self, x0, t, eta=None):
        """
        x0: (b, c, h, w), original data
        t: (b,), time step (0 <= t < n_steps)
        """
        b, c, h, w = x0.shape
        if eta is None:
            eta = torch.randn(b, c, h, w, device=x0.device)

        alpha_bar = self.alpha_bars[t]
        noisy_x = alpha_bar.sqrt().reshape(b, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(b, 1, 1, 1) * eta

        return noisy_x

    def denoise(self, xt, t, cond):
        """
        xt: (b, in_c, h, w), noisy input (concatenated with some data)
        cond: (b, cond_nc, h, w), conditional input
        t: (b,), time step (0 <= t < n_steps)
        """
        pred_noise = self(xt, t, cond)
        return pred_noise

    @torch.no_grad()
    def _build_progress_iter(self, iterable, total, mode: str):
        """
        Internal helper to create a progress iterator based on verbose mode.
        """
        mode = (mode or "none").lower()
        if mode == "tqdm":
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=total, desc="DDPM sampling", leave=False), mode
            except Exception:
                return iterable, "none"
        return iterable, mode

    @torch.no_grad()
    def sample_ddpm(self, cond, input_cond=None, verbose: str = "none", store_intermediate: bool = False):
        """
        input_frame: (b, c, h, w) number of input frames (conditional input frames) for the diffusion model
        cond: (b, cond_nc, h, w), conditional input
        verbose: "none", "text", or "tqdm" for progress display
        """
        ## confirm that the model is in eval mode
        self.backbone_model.eval()
        
        B, C, H, W = cond.shape
        ## get cond device
        device = cond.device

        x = torch.randn(B, *self.output_shape, device=device)

        progress_iter_raw = reversed(range(self.n_steps))
        progress_iter, mode = self._build_progress_iter(progress_iter_raw, self.n_steps, verbose)
        use_text = mode == "text"

        text_interval = max(1, self.n_steps // 10)
        
        frames = []
        for idx, t in enumerate(progress_iter):
            time_tensor = (torch.ones(B, device=device) * t).long()
            if input_cond is not None:
                input_ = torch.cat((x, input_cond), dim=1)
            else:
                input_ = x

            eta_theta = self.denoise(input_, time_tensor, cond)

            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]

            a = 1 / alpha_t.sqrt()
            b = ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * eta_theta

            x = a * (x - b)
            if t > 0:
                z = torch.randn(B, *self.output_shape, device=device)
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            ## store intermediate frames for visualization
            if (idx % 50 == 0) or (t == 0):
                out = x.clone()
                out = ((out + 1) / 2).clamp(0, 1)
                out = out.cpu().numpy()
                frames.append(out)
            
            if use_text and (idx + 1) % text_interval == 0:
                print(f"DDPM sampling {idx + 1}/{self.n_steps}", flush=True)

        if mode == "tqdm" and hasattr(progress_iter, "close"):
            progress_iter.close()
        
        if store_intermediate:
            return x, frames
        else:
            return x

    @torch.no_grad()
    def sample_ddim(self, cond, input_cond=None, ddim_steps: int = 100, eta: float = 0.2, verbose: str = "none", store_intermediate: bool = False):
        """
        Deterministic/stochastic DDIM sampling.

        cond: (b, cond_nc, h, w)
        input_cond: optional conditional input concatenated with the predicted frames
        ddim_steps: number of steps to sample (<= n_steps)
        eta: 0 for deterministic DDIM, >0 adds noise controlled by eta
        verbose: "none", "text", or "tqdm" for progress display
        """
        self.backbone_model.eval()

        B, C, H, W = cond.shape
        device = cond.device
        ddim_steps = max(1, min(ddim_steps, self.n_steps))

        # create evenly spaced timesteps
        ddim_timesteps = torch.linspace(0, self.n_steps - 1, steps=ddim_steps, device=device).long()
        ddim_timesteps = torch.unique(ddim_timesteps, sorted=True)  # safety against duplicates
        ddim_t_reverse = list(reversed(ddim_timesteps.tolist()))

        x = torch.randn(B, *self.output_shape, device=device)

        progress_iter_raw = enumerate(ddim_t_reverse)
        progress_iter, mode = self._build_progress_iter(progress_iter_raw, len(ddim_t_reverse), verbose)
        use_text = mode == "text"
        text_interval = max(1, len(ddim_t_reverse) // 10)

        frames = []
        for idx, (iter_idx, t) in enumerate(progress_iter):
            time_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            if input_cond is not None:
                input_ = torch.cat((x, input_cond), dim=1)
            else:
                input_ = x

            eps = self.denoise(input_, time_tensor, cond)

            alpha_bar_t = self.alpha_bars[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

            if iter_idx + 1 < len(ddim_t_reverse):
                t_prev = ddim_t_reverse[iter_idx + 1]
                alpha_bar_prev = self.alpha_bars[t_prev]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t, device=device)

            sigma_t = 0.0
            if eta > 0 and alpha_bar_prev < 1:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
                )

            sigma_t = torch.as_tensor(sigma_t, device=device, dtype=x.dtype)
            noise = torch.randn_like(x) if (eta > 0 and alpha_bar_prev < 1) else torch.zeros_like(x)

            c_t = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0.0))
            x = (
                alpha_bar_prev.sqrt() * x0_pred
                + c_t * eps
                + sigma_t * noise
            )

            ## store intermediate frames for visualization
            if (idx % 25 == 0) or (t == 0):
                out = x.clone()
                out = ((out + 1) / 2).clamp(0, 1)
                out = out.cpu().numpy()
                frames.append(out)

            if use_text and (idx + 1) % text_interval == 0:
                print(f"DDIM sampling {idx + 1}/{len(ddim_t_reverse)}", flush=True)

        if mode == "tqdm" and hasattr(progress_iter, "close"):
            progress_iter.close()

        if store_intermediate:
            return x, frames
        else:
            return x

    # Backward-compatible alias
    sample = sample_ddpm
