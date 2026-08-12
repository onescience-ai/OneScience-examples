"""Train compact MetNet-3 with multi-task epochs, validation and resume."""

import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from model import MetNet3, MetNet3Config, multitask_loss, validate_batch
from model.fake_data import FakeMetNetDataset, make_fake

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--train-samples",type=int,default=32); p.add_argument("--validation-samples",type=int,default=8); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--patience",type=int,default=3); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--checkpoint-dir",type=Path,default=ROOT/"weight/training"); p.add_argument("--resume",type=Path); a=p.parse_args()
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); a.checkpoint_dir.mkdir(parents=True,exist_ok=True); device=torch.device(a.device); config=MetNet3Config(); model=MetNet3(config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=max(1,a.patience//2),factor=.5)
    train=DataLoader(FakeMetNetDataset(config,a.train_samples,a.seed),batch_size=a.batch_size,shuffle=True); val=DataLoader(FakeMetNetDataset(config,a.validation_samples,a.seed+100000),batch_size=a.batch_size); start=0; best=float("inf"); history=[]
    if a.resume:
        q=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(q["model"]); opt.load_state_dict(q["optimizer"]); sched.load_state_dict(q["scheduler"]); start=q["epoch"]+1; best=q["best_val_loss"]; history=q["history"]
    stale=0
    for epoch in range(start,a.epochs):
        model.train(); tr=[]
        for batch,target in train:
            batch={k:v.to(device) for k,v in batch.items()}; target={k:v.to(device) for k,v in target.items()}; validate_batch(batch,config); opt.zero_grad(set_to_none=True); total,_=multitask_loss(model(batch),target); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr.append(float(total.detach()))
        model.eval(); va=[]
        with torch.inference_mode():
            for batch,target in val:
                batch={k:v.to(device) for k,v in batch.items()}; target={k:v.to(device) for k,v in target.items()}; va.append(float(multitask_loss(model(batch),target)[0]))
        loss=float(np.mean(va)); sched.step(loss); row={"epoch":epoch,"train_loss":float(np.mean(tr)),"validation_loss":loss,"learning_rate":opt.param_groups[0]["lr"]}; history.append(row); improved=loss<best; best=min(best,loss); stale=0 if improved else stale+1
        train_config={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}; payload={"model":model.state_dict(),"config":config.__dict__,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"epoch":epoch,"global_step":(epoch+1)*len(train),"best_val_loss":best,"history":history,"train_config":train_config}; torch.save(payload,a.checkpoint_dir/"latest.pth"); torch.save(payload,ROOT/"weight/model.pth");
        if improved: torch.save(payload,a.checkpoint_dir/"best.pth")
        (a.checkpoint_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n"); print(json.dumps(row))
        if stale>=a.patience: break
    print(json.dumps({"status":"completed","epochs_completed":len(history),"best_validation_loss":best},indent=2))

if __name__=="__main__": main()
