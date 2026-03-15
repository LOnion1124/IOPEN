from src.eval import make_evaluator
from src.datasets import make_dataloader

dataloader = make_dataloader()
batch = next(iter(dataloader))

evaluator = make_evaluator()
evaluator.evaluate(batch)