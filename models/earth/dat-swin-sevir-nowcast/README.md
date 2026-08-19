Minimal smoke-test bundle for:
https://huggingface.co/huilinsigehigh/dat-swin-sevir-nowcast

Files:
- run_smoke.py: self-contained Swin-T + UNet nowcasting smoke test with synthetic data
- requirements_smoke.txt: only torch + torchvision

Run:
  python3 -m pip install -r requirements_smoke.txt
  python3 run_smoke.py

Success indicator:
  SMOKE TEST PASSED

The script uses no SEVIR dataset and no pretrained weights. It creates a tiny
1 x 12-frame x 64 x 64 moving-Gaussian sequence internally, uses the first
6 frames as input and the last 6 as target, performs one forward pass,
one backward pass, and one AdamW optimizer step.
