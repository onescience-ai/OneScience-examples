"""
Weights & Biases utilities for VGB-DM.
"""
import wandb

class wandb_util:
    """Wrapper for wandb utility functions."""
    
    @staticmethod
    def init_wandb(exp_config=None, **kwargs):
        """Initialize wandb run and return the exp_config."""
        # 如果传入了 exp_config，使用它；否则使用 kwargs 中的 config
        config = exp_config or kwargs.get('config', {})
        project = kwargs.get('project', 'VGB-DM')
        name = kwargs.get('name', None)
        mode = kwargs.get('mode', 'offline')
        
        # 初始化 wandb
        if mode != 'disabled':
            wandb.init(project=project, name=name, config=config, mode=mode)
        else:
            print("WandB is disabled (mode=disabled)")
        
        # 返回 exp_config（确保它是字典）
        if exp_config is not None:
            return exp_config
        return config
    
    @staticmethod
    def log_metrics(metrics, step=None, **kwargs):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)
    
    @staticmethod
    def finish_wandb():
        """Finish wandb run."""
        wandb.finish()
    
    @staticmethod
    def watch(model, **kwargs):
        """Watch model gradients and parameters."""
        wandb.watch(model, **kwargs)
