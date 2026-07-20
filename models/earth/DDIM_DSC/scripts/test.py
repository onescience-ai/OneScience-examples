#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from diffusers import DiffusionPipeline


# In[2]:


# load the custom DDIM pipeline
pipe = DiffusionPipeline.from_pretrained(
    "lschmidt/ddim-dsc",
    custom_pipeline="cond_ddim_pipeline",
    trust_remote_code=True
)


# In[3]:


# create a sample low-resolution input --> shape: (sequence_length, channels, height, width)
lres_image = torch.randn((3, 6, 32, 32)).to(pipe.device)


# In[4]:


# Run inference and get output object
import numpy as np
from PIL import Image
outputs = pipe(image=lres_image, output_type="np")

# Check object structure
print(f"Output object type: {type(outputs)}")
print(f"Output object attributes: {dir(outputs)}")

# Usually diffusers uses the 'images' attribute
if hasattr(outputs, 'images'):
    output_tensor = outputs.images
    print(f"Output shape: {output_tensor.shape}")
    print(f"Output dtype: {output_tensor.dtype}")
else:
    raise AttributeError("Output object does not contain 'images' attribute")


# In[5]:


import numpy as np
from PIL import Image
import os

# Create save directory
save_dir = "output_images"
os.makedirs(save_dir, exist_ok=True)

# Iterate over all samples and channels
for sample_idx in range(output_tensor.shape[0]):      # Iterate over batch dimension
    for ch_idx in range(output_tensor.shape[1]):      # Iterate over channel dimension
        # Get single image data
        image_data = output_tensor[sample_idx, ch_idx, :, :]
        
        # Normalize to 0-255 and convert to uint8
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8) * 255
        image_data = image_data.astype(np.uint8)
        
        # Channel name
        channel_name = "u" if ch_idx == 0 else "v"
        
        # Save as image
        img = Image.fromarray(image_data, mode='L')
        filename = f"{save_dir}/sample_{sample_idx}_{channel_name}.png"
        img.save(filename)
        print(f"✅ Saved: {filename}")

print(f"\n✅ Total {output_tensor.shape[0] * output_tensor.shape[1]} images saved to {save_dir}/")


# In[7]:


# ==================== Evaluation Metrics ====================
print("\n" + "=" * 60)
print("📊 EVALUATION METRICS")
print("=" * 60)

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

# Helper function to normalize image to [0, 1]
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

# ===== 1. Pixel-level Metrics: PSNR and SSIM =====
print("\n🔹 Pixel-level Metrics (PSNR & SSIM)")
print("-" * 40)

psnr_list = []
ssim_list = []

for sample_idx in range(output_tensor.shape[0]):
    for ch_idx in range(output_tensor.shape[1]):
        # Get generated output
        sr_image = output_tensor[sample_idx, ch_idx, :, :]
        
        # Get corresponding input (low-resolution)
        if hasattr(lres_image, 'cpu'):
            lr_image = lres_image.cpu().numpy()[sample_idx, ch_idx, :, :]
        else:
            lr_image = lres_image[sample_idx, ch_idx, :, :]
        
        # Normalize both images to [0, 1]
        sr_norm = normalize(sr_image)
        lr_norm = normalize(lr_image)
        
        # Compute PSNR (data_range=1.0 since images are in [0, 1])
        psnr_val = psnr(lr_norm, sr_norm, data_range=1.0)
        
        # Compute SSIM
        ssim_val = ssim(sr_norm, lr_norm, data_range=1.0)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        channel_name = "u" if ch_idx == 0 else "v"
        print(f"  Sample {sample_idx}, {channel_name}: PSNR = {psnr_val:.4f} dB, SSIM = {ssim_val:.4f}")

# Average metrics
print("-" * 40)
print(f"📊 Average PSNR: {np.mean(psnr_list):.4f} dB")
print(f"📊 Average SSIM: {np.mean(ssim_list):.4f}")
print("💡 Higher PSNR and SSIM closer to 1 indicate better quality")

# ===== 2. Perceptual Metrics: LPIPS =====
print("\n🔹 Perceptual Metric (LPIPS)")
print("-" * 40)

try:
    import lpips
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex', verbose=False)
    
    lpips_list = []
    
    for sample_idx in range(output_tensor.shape[0]):
        for ch_idx in range(output_tensor.shape[1]):
            # Get images as tensors (LPIPS expects PyTorch tensors)
            sr_tensor = torch.from_numpy(output_tensor[sample_idx, ch_idx, :, :]).unsqueeze(0).unsqueeze(0).float()
            if hasattr(lres_image, 'cpu'):
                lr_tensor = lres_image.cpu()[sample_idx, ch_idx, :, :].unsqueeze(0).unsqueeze(0).float()
            else:
                lr_tensor = torch.from_numpy(lres_image[sample_idx, ch_idx, :, :]).unsqueeze(0).unsqueeze(0).float()
            
            # Normalize to [0, 1] (LPIPS expects [-1, 1] or [0, 1])
            sr_tensor = (sr_tensor - sr_tensor.min()) / (sr_tensor.max() - sr_tensor.min() + 1e-8)
            lr_tensor = (lr_tensor - lr_tensor.min()) / (lr_tensor.max() - lr_tensor.min() + 1e-8)
            
            # Compute LPIPS (lower is better)
            lpips_val = lpips_model(sr_tensor, lr_tensor).item()
            lpips_list.append(lpips_val)
            
            channel_name = "u" if ch_idx == 0 else "v"
            print(f"  Sample {sample_idx}, {channel_name}: LPIPS = {lpips_val:.4f}")
    
    print("-" * 40)
    print(f"📊 Average LPIPS: {np.mean(lpips_list):.4f}")
    print("💡 Lower LPIPS indicates better perceptual similarity")
    
except ImportError:
    print("⚠️  LPIPS not installed. Run: pip install lpips")
except Exception as e:
    print(f"⚠️  LPIPS computation failed: {e}")

# ===== 3. Summary =====
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"✅ Total images evaluated: {output_tensor.shape[0] * output_tensor.shape[1]}")
print(f"✅ Average PSNR:  {np.mean(psnr_list):.4f} dB")
print(f"✅ Average SSIM:  {np.mean(ssim_list):.4f}")
if 'lpips_list' in locals() and len(lpips_list) > 0:
    print(f"✅ Average LPIPS: {np.mean(lpips_list):.4f}")
print("=" * 60)


# In[ ]:




