import os
import logging

import torch
from tensorboardX import SummaryWriter

from module.argparser import ArgParser
from module.dataset.factory import create_train_valid_dataset
from module.model.factory import create_model
from module.optimizer.factory import create_optimizer
from module.loss.factory import create_criterions
from module.metrics.factory import create_metrics
from module import Trainer
from module import set_random_seed

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device use to train {}".format(device))

    # Preparing arguments
    parser = ArgParser()
    args = parser.parse()

    # Set random seed
    set_random_seed(args.random_seed)

    # Preparing tensorboardX writer
    if args.model_dir is None:
        raise Exception("Model directory is not given.")
    if not os.path.exists(args.model_dir):
        logging.info("creating model directory {}...".format(args.model_dir))
        os.makedirs(args.model_dir)
    writer = SummaryWriter(os.path.join(args.model_dir, "train_logs"))

    # Preparing dataset
    if args.data_dir is None:
        raise Exception("Data directory is not given.")
    data = create_train_valid_dataset(args)
    num_class_1 = data["trainset"].num_class_1
    num_class_2 = data["trainset"].num_class_2

    # Preparing model
    model = create_model(args, num_class_1, num_class_2)
    model.to(device)

    # Preparing optimizer
    optimizer = create_optimizer(args, model)

    # Preparing criterions
    criterions = {
        "class_1": create_criterions(args, num_class_1, device),
        "class_2": create_criterions(args, num_class_2, device),
    }

    # Preparing metrics
    metrics = {"class_1": create_metrics(args), "class_2": create_metrics(args)}

    # Preparing trainer
    trainer = Trainer(
        model=model,
        optim=optimizer,
        criterions=criterions,
        metric=metrics,
        scheduler=None,
        train_dl=data["train_dataloader"],
        val_dl=data["valid_dataloader"],
        writer=writer,
        save_dir=args.model_dir,
        device=device,
    )
    trainer.fit(args.epochs)
