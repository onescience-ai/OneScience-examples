#!/usr/bin/env python3
"""Train SFNO with pair-wise epochs, validation and resumable checkpoints."""

from __future__ import annotations

import argparse, importlib.metadata, json, math, random, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; LOCAL_DEPS=ROOT/".deps"
if LOCAL_DEPS.is_dir(): sys.path.insert(0,str(LOCAL_DEPS))
sys.path.insert(0,str(ROOT))
from model.config import load_config
from model.dataset import SphericalPairDataset
from model.fake_spherical_data import make_fake_spherical_sequence
from model.sfno_adapter import OfficialSFNOAdapter

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=ROOT/"model/default_config.json"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int); p.add_argument("--learning-rate",type=float); p.add_argument("--validation-fraction",type=float,default=.25); p.add_argument("--patience",type=int,default=3); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--checkpoint-dir",type=Path,default=ROOT/"weight/training"); p.add_argument("--resume",type=Path); a=p.parse_args(); config=load_config(a.config)
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed); device=torch.device(a.device); a.checkpoint_dir.mkdir(parents=True,exist_ok=True); fields=make_fake_spherical_sequence(config.timesteps,config.channels,config.nlat,config.nlon,config.seed)["fields"]
    if not 0.0 < a.validation_fraction < 1.0: raise ValueError("validation_fraction must be between 0 and 1")
    split=max(1,int((config.timesteps-1)*(1-a.validation_fraction))); val_start=min(split,config.timesteps-2); train=DataLoader(SphericalPairDataset(fields,0,val_start),batch_size=a.batch_size or config.batch_size,shuffle=True); val=DataLoader(SphericalPairDataset(fields,val_start,config.timesteps-1),batch_size=a.batch_size or config.batch_size)
    model=OfficialSFNOAdapter(config).to(device); opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate or config.learning_rate); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=max(1,a.patience//2),factor=.5); start=0; best=float("inf"); history=[]
    if a.resume:
        q=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(q["model"]); opt.load_state_dict(q["optimizer"]); sched.load_state_dict(q["scheduler"]); start=q["epoch"]+1; best=q["best_val_loss"]; history=q["history"]
    stale=0
    for epoch in range(start,a.epochs):
        model.train(); tr=[]
        for x,y in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr.append(float(loss.detach()))
        model.eval(); va=[]
        with torch.inference_mode():
            for x,y in val: va.append(float(torch.nn.functional.mse_loss(model(x.to(device)),y.to(device))))
        loss=float(np.mean(va)); sched.step(loss); row={"epoch":epoch,"train_loss":float(np.mean(tr)),"validation_loss":loss,"learning_rate":opt.param_groups[0]["lr"]}; history.append(row); improved=loss<best; best=min(best,loss); stale=0 if improved else stale+1
        train_config={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}; payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":vars(config),"epoch":epoch,"global_step":(epoch+1)*len(train),"best_val_loss":best,"history":history,"train_config":train_config,"torch_harmonics_version":importlib.metadata.version("torch-harmonics")}; torch.save(payload,a.checkpoint_dir/"latest.pth"); torch.save(payload,ROOT/"weight/model.pth");
        if improved: torch.save(payload,a.checkpoint_dir/"best.pth")
        (a.checkpoint_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n"); print(json.dumps(row))
        if stale>=a.patience: break
    print(json.dumps({"status":"completed","epochs_completed":len(history),"best_validation_loss":best},indent=2))
if __name__=="__main__": main()
