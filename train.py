from src.config import cfg, args
from src.train import make_trainer

train_cfg = cfg['train']

trainer = make_trainer(
    epochs=train_cfg['epoch'],
    log_interval=train_cfg['log_interval'],
    ckpt_dir=train_cfg['result_path'],
    save_interval=train_cfg['save_interval'],
    temperature=train_cfg.get('loss_temperature', 0.1),
    alpha=train_cfg.get('loss_alpha', 10.0),
    use_adaptive_weight=train_cfg['use_adaptive_weight'],
    coarse_only_epochs=train_cfg.get('coarse_only_epochs', 5),
    early_stopping_enabled=train_cfg.get('early_stopping_enabled', True),
    early_stopping_patience=train_cfg.get('early_stopping_patience', 10),
    early_stopping_min_delta=train_cfg.get('early_stopping_min_delta', 1e-4),
    )

if train_cfg['use_checkpoint']:
    trainer.load_checkpoint(train_cfg['checkpoint_path'])
    print(f"Loaded {train_cfg['checkpoint_path']}")

trainer.train()