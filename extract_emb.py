import os
import json
import logging

import torch
import numpy as np
from tqdm import tqdm

from module.argparser import ArgParser
from module.dataset.factory import create_train_valid_dataset
from module.model.factory import create_model
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

    # Check whether embedding directory is given
    if args.embedding_path is None:
        embeds_path = os.path.join(".", "embedding.json")
        logging.info(
            f"embedding store path is not given, extracted embedding will be stored in "
            f"default path: {embeds_path}"
        )
    else:
        embeds_path = args.embedding_path

    # Preparing dataset
    if args.data_dir is None:
        raise Exception("Data directory is not given.")
    data = create_train_valid_dataset(args, mode="test")
    num_progtype = data["trainset"].num_progtype
    num_ch_codename = data["trainset"].num_ch_codename

    # Preparing model
    if args.model is None:
        raise Exception("Model weight is not given.")
    model = create_model(args, num_progtype, num_ch_codename)
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    eventids = []
    embs = []

    trange = tqdm(data["test_dataloader"], total=len(data["test_dataloader"]))
    with torch.no_grad():
        for batch in trange:
            eventids += batch["eventid"]

            # prepare training data
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if v is not None and k != "id" and not isinstance(v, list)
            }

            # extract embeddings
            emb = model.extract_feature(batch)
            embs.append(emb.cpu().numpy())

    embs = np.concatenate(embs)

    with open(embeds_path, "w") as fout:
        output_dict = {str(id): emb.tolist() for id, emb in zip(eventids, embs)}
        json.dump(output_dict, fout)

    logging.info(f"extracted embedding will be stored in {embeds_path}")
