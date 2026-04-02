from src.eval import make_evaluator
from src.datasets import make_dataloader
from src.config import cfg, args

eval_mode = cfg['eval']['mode']
evaluator = make_evaluator()

if eval_mode == 'batch':
    val_batch_size = cfg['train'].get('val_batch_size', cfg['train']['batch_size'])
    dataloader = make_dataloader(split='validate', shuffle=False, batch_size=val_batch_size)
    batch = next(iter(dataloader))
    evaluator.evaluate(mode='batch', batch=batch)
elif eval_mode == 'coco':
    evaluator.evaluate(mode='coco')