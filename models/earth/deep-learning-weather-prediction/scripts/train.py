"""Train the compact DLWP-CS model with epochs, validation and resume."""

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import DLWPCubeSphereUNet, CubeSpherePadding2d, capped_leaky_relu, weighted_mse
from model.dataset import FakeCubeSphereDataset

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--train-samples",type=int,default=64); p.add_argument("--validation-samples",type=int,default=16); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=0.0); p.add_argument("--patience",type=int,default=3); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--checkpoint-dir",type=Path,default=ROOT/"weight/training"); p.add_argument("--resume",type=Path); return p.parse_args()

def main():
    a=parse_args(); device=torch.device(a.device); 
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); a.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    train=DataLoader(FakeCubeSphereDataset(a.train_samples,a.seed),batch_size=a.batch_size,shuffle=True); val=DataLoader(FakeCubeSphereDataset(a.validation_samples,a.seed+100000),batch_size=a.batch_size)
    model=DLWPCubeSphereUNet(2,2,base_channels=4).to(device); opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=max(1,a.patience//2),factor=.5)
    start=0; best=float("inf"); history=[]
    if a.resume:
        q=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(q["model"]); opt.load_state_dict(q["optimizer"]); sched.load_state_dict(q["scheduler"]); start=q["epoch"]+1; best=q["best_val_loss"]; history=q["history"]
    stale=0
    for epoch in range(start,a.epochs):
        model.train(); train_losses=[]
        for x in train:
            x=x.to(device); opt.zero_grad(set_to_none=True); pred=model(x); loss=weighted_mse(pred,x); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); train_losses.append(float(loss.detach()))
        model.eval(); val_losses=[]
        with torch.inference_mode():
            for x in val:
                x=x.to(device); val_losses.append(float(weighted_mse(model(x),x)))
        va=float(np.mean(val_losses)); sched.step(va); row={"epoch":epoch,"train_loss":float(np.mean(train_losses)),"validation_loss":va,"learning_rate":opt.param_groups[0]["lr"]}; history.append(row); improved=va<best; best=min(best,va); stale=0 if improved else stale+1
        train_config={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"global_step":(epoch+1)*len(train),"best_val_loss":best,"history":history,"model_config":{"in_channels":2,"out_channels":2,"base_channels":4},"train_config":train_config}
        torch.save(payload,a.checkpoint_dir/"latest.pth"); torch.save(payload,ROOT/"weight/model.pth")
        if improved: torch.save(payload,a.checkpoint_dir/"best.pth")
        (a.checkpoint_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n"); print(json.dumps(row))
        if stale>=a.patience: break
    print(json.dumps({"status":"completed","epochs_completed":len(history),"best_validation_loss":best},indent=2))

if __name__=="__main__": main()
