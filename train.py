from src.config import cfg, args
from src.train import make_trainer

train_cfg = cfg['train']
geom_cfg = train_cfg.get('geometry_regularization', {})

trainer = make_trainer(
    epochs=train_cfg['epoch'],
    log_interval=train_cfg['log_interval'],
    ckpt_dir=train_cfg['result_path'],
    save_interval=train_cfg['save_interval'],
    temperature=train_cfg.get('loss_temperature', 0.1),
    alpha=train_cfg.get('loss_alpha', 10.0),
    use_adaptive_weight=train_cfg['use_adaptive_weight'],
    coarse_only_epochs=train_cfg.get('coarse_only_epochs', 5),
    geom_reg_enabled=geom_cfg.get('enabled', False),
    geom_reg_weight=geom_cfg.get('weight', 0.0),
    geom_equal_length_weight=geom_cfg.get('equal_length_weight', 1.0),
    geom_orthogonality_weight=geom_cfg.get('orthogonality_weight', 1.0),
    geom_parallel_weight=geom_cfg.get('parallel_weight', 1.0),
    geom_eps=geom_cfg.get('eps', 1e-6),
    early_stopping_enabled=train_cfg.get('early_stopping_enabled', True),
    early_stopping_patience=train_cfg.get('early_stopping_patience', 10),
    early_stopping_min_delta=train_cfg.get('early_stopping_min_delta', 1e-4),
    )

if train_cfg['use_checkpoint']:
    trainer.load_checkpoint(train_cfg['checkpoint_path'])
    print(f"Loaded {train_cfg['checkpoint_path']}")

trainer.train()