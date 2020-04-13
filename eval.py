import logging

import torch
from tqdm import tqdm

from module.argparser import ArgParser
from module.dataset.factory import create_train_valid_dataset
from module.model.factory import create_model
from module.metrics.factory import create_metrics
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

    # Preparing dataset
    if args.data_dir is None:
        raise Exception("Data directory is not given.")
    data = create_train_valid_dataset(args)
    num_progtype = data["trainset"].num_progtype
    num_ch_codename = data["trainset"].num_ch_codename

    # Preparing model
    if args.model is None:
        raise Exception("Model weight is not given.")
    model = create_model(args, num_progtype, num_ch_codename)
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    # Preparing metrics
    metrics = {"class_1": create_metrics(args), "class_2": create_metrics(args)}

    for _, metric in metrics.items():
        metric.reset()

    trange = tqdm(data["valid_dataloader"], total=len(data["valid_dataloader"]))
    with torch.no_grad():
        for batch in trange:
            # prepare training data
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if v is not None and k != "id" and not isinstance(v, list)
            }

            # calculate label loss
            items = model(batch)

            # update metric
            metrics["class_1"].update(
                items["class_1"].cpu().numpy(), batch["class_1"].cpu().numpy()
            )
            metrics["class_2"].update(
                items["class_2"].cpu().numpy(), batch["class_2"].cpu().numpy()
            )

    logging.info(
        "===> Model Accuracy: class_1 {:.5f}, class_2 {:.5f}, average {:.5f}.".format(
            metrics["class_1"].get_score(),
            metrics["class_2"].get_score(),
            (metrics["class_1"].get_score() + metrics["class_2"].get_score()) / 2,
        )
    )
