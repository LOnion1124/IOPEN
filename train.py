from src.config import cfg, args
from src.train import make_trainer

train_cfg = cfg['train']

trainer = make_trainer(
    epochs=train_cfg['epoch'],
    log_interval=50,
    ckpt_dir=train_cfg['result_path'],
    save_interval=25,
    use_adaptive_weight=train_cfg['use_adaptive_weight']
    )
trainer.train()