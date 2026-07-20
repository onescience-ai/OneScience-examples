#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
from tqdm import tqdm
from pathlib import Path

# Model repository ID
REPO_ID = "rwiegels/seasonal_afnocast"
# Base URL for building file download links
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"

# Local save directory - current directory
SAVE_DIR = Path(".")

# List of filenames to download
# Based on your screenshot, generates afnocast_best_01 through afnocast_best_12
file_names = [f"afnocast_best_{i:02d}.safetensors" for i in range(1, 13)]

def download_file(url, local_path):
    """Download a single file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if request was successful

        # Get total file size (bytes)
        total_size = int(response.headers.get('content-length', 0))
        # Use tqdm to show progress
        with open(local_path, 'wb') as file, tqdm(
            desc=local_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"✅ Download complete: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed {url}: {e}")
        return False

# Main download loop
print(f"🚀 Starting download of {len(file_names)} weight files to current directory: {os.getcwd()}")
for name in file_names:
    file_url = BASE_URL + name
    local_file_path = SAVE_DIR / name
    print(f"⬇️  Downloading: {name}")
    success = download_file(file_url, local_file_path)
    if not success:
        print(f"Please check if file '{name}' exists in the repository, or check your network connection.")

print("🏁 All download tasks completed.")

