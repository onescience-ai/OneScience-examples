"""Train tiny AtmoRep with train/validation epochs and resumable checkpoints."""

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from model.fake_data import FakeAtmoRepDataset
from model.tiny_atmorep import TinyAtmoRep, TinyAtmoRepConfig, ensemble_statistical_loss


def args():
    p0=argparse.ArgumentParser(add_help=False); p0.add_argument("--config",type=Path,default=ROOT/"conf/config.yaml"); k,_=p0.parse_known_args(); c=yaml.safe_load(k.config.read_text())["training"]
    p=argparse.ArgumentParser(parents=[p0]);
    for name, typ in (("epochs", int), ("batch-size", int), ("train-samples", int), ("validation-samples", int), ("seed", int), ("patience", int), ("num-workers", int), ("learning-rate", float), ("weight-decay", float), ("mask-fraction", float)):
        p.add_argument("--" + name, type=typ, default=c[name.replace("-", "_")])
    p.add_argument("--checkpoint-dir",type=Path,default=ROOT/c["checkpoint_dir"]); p.add_argument("--resume",type=Path); p.add_argument("--device",choices=("cpu","cuda"),default=c["device"]); return p.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None); totals=[]
    context=torch.enable_grad() if optimizer else torch.inference_mode()
    with context:
        for fields,mask in loader:
            fields,mask=fields.to(device),mask.to(device); pred=model(fields,mask,level=137.0); target=model.tokenize(fields); loss,_=ensemble_statistical_loss(pred,target,mask)
            if optimizer: optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            totals.append(float(loss.detach()))
    return float(np.mean(totals))


def main():
    a=args(); device=torch.device(a.device); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); a.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    config=TinyAtmoRepConfig(); model=TinyAtmoRep(config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=max(1,a.patience//2),factor=.5)
    train=DataLoader(FakeAtmoRepDataset(config,a.train_samples,a.seed,a.mask_fraction),batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers); val=DataLoader(FakeAtmoRepDataset(config,a.validation_samples,a.seed+100000,a.mask_fraction),batch_size=a.batch_size,num_workers=a.num_workers)
    start=0; best=float("inf"); history=[]
    if a.resume:
        q=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(q["model"]); opt.load_state_dict(q["optimizer"]); sched.load_state_dict(q["scheduler"]); start=q["epoch"]+1; best=q["best_val_loss"]; history=q["history"]
    stale=0
    for epoch in range(start,a.epochs):
        tr=run_epoch(model,train,device,opt); va=run_epoch(model,val,device); sched.step(va); row={"epoch":epoch,"train_loss":tr,"validation_loss":va,"learning_rate":opt.param_groups[0]["lr"]}; history.append(row); improved=va<best; best=min(best,va); stale=0 if improved else stale+1
        train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":config.to_dict(),"epoch":epoch,"step":len(train)*(epoch+1),"best_val_loss":best,"history":history,"train_config":train_config}
        torch.save(payload,a.checkpoint_dir/"latest.pth"); torch.save(payload,ROOT/"weight/tiny_atmorep.pth")
        if improved: torch.save(payload,a.checkpoint_dir/"best.pth")
        (a.checkpoint_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n"); print(json.dumps(row))
        if stale>=a.patience: break
    print(json.dumps({"status":"completed","epochs_completed":len(history),"best_validation_loss":best},indent=2))
if __name__=="__main__": main()
