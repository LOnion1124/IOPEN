from src.config import cfg, args
from src.train import make_trainer

train_cfg = cfg['train']

trainer = make_trainer(
    epochs=train_cfg['epoch'],
    log_interval=train_cfg['log_interval'],
    ckpt_dir=train_cfg['result_path'],
    save_interval=train_cfg['save_interval'],
    use_adaptive_weight=train_cfg['use_adaptive_weight']
    )

if train_cfg['use_checkpoint']:
    trainer.load_checkpoint(train_cfg['checkpoint_path'])
    print(f"Loaded {train_cfg['checkpoint_path']}")

trainer.train()