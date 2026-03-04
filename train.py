from src.config import cfg, args
from src.train import make_trainer

train_cfg = cfg['train']

trainer = make_trainer(
    epochs=train_cfg['epoch'],
    log_interval=10,
    ckpt_dir=train_cfg['result_path'],
    save_interval=1
    )
trainer.train()