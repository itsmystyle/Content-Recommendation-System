import torch.optim as optim
from transformers import AdamW

from module.optimizer import MultipleOptimizer


def create_optimizer(args, model):
    if args.multi_optimizer:
        optimizer = MultipleOptimizer(
            AdamW(
                [{"params": model.bert.parameters()}], lr=args.learning_rate, eps=args.adam_epsilon
            ),
            optim.Adam(
                [{"params": i.parameters()} for i in model.children()][1:],
                lr=1e-4,
                weight_decay=args.weight_decay,
            ),
        )
    else:
        optimizer = MultipleOptimizer(
            AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        )

    return optimizer
