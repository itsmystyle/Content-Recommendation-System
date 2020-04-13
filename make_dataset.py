import os
import pickle
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from module import ArgParser
from module.utils import set_random_seed


def make_dataset(args):
    if args.data is None:
        raise Exception("Data path is not given.")
    if args.data_dir is None:
        raise Exception("Data directory path is not given.")

    df = pd.read_csv(args.data)
    logging.info("number of data (duplicated data included) {}...".format(df.shape[0]))
    if args.remove_duplicate:
        df = df.drop_duplicates(["programname_descr"])
        logging.info("number of data after duplicated data is removed {}...".format(df.shape[0]))

    # proprocessing
    # 1. Filling nan columns with 'UNK' for class_1 and class_2
    df = df.fillna("UNK")
    df["labels"] = df.apply(lambda x: "$$$".join([x.class_1, x.class_2]), axis=1)

    if not os.path.exists(args.data_dir):
        logging.info("creating data directory {}...".format(args.data_dir))
        os.makedirs(args.data_dir)

    # creating class_1 mapping and class_2 mapping
    class_1 = sorted(df.class_1.unique())
    class_1_idx2item = {idx: v for idx, v in enumerate(class_1)}
    class_1_item2idx = {v: k for k, v in class_1_idx2item.items()}
    class_1_dict = {"idx2item": class_1_idx2item, "item2idx": class_1_item2idx}
    class_1_mapping_path = os.path.join(args.data_dir, "class_1_mapping.pkl")
    logging.info("class_1 mapping save to {}".format(class_1_mapping_path))
    with open(class_1_mapping_path, "wb") as fout:
        pickle.dump(class_1_dict, fout)

    class_2 = sorted(df.class_2.unique())
    class_2_idx2item = {idx: v for idx, v in enumerate(class_2)}
    class_2_item2idx = {v: k for k, v in class_2_idx2item.items()}
    class_2_dict = {"idx2item": class_2_idx2item, "item2idx": class_2_item2idx}
    class_2_path = os.path.join(args.data_dir, "class_2_mapping.pkl")
    logging.info("class_2 mapping save to {}".format(class_2_path))
    with open(class_2_path, "wb") as fout:
        pickle.dump(class_2_dict, fout)

    if args.valid:
        logging.info(
            "creating training and development set with split ratio of {}...".format(
                args.split_ratio
            )
        )
        val_cnt = df.labels.value_counts()
        left_X = df[df.labels.isin(val_cnt[val_cnt == 1].index)]
        X = df[~df.labels.isin(val_cnt[val_cnt == 1].index)]
        X_train, X_test = train_test_split(
            X, test_size=args.split_ratio, random_state=args.random_seed, stratify=X.labels
        )
        X_train = pd.concat([X_train, left_X], axis=0)

        logging.info(
            "number of training data {}, number of validation data {}...".format(
                X_train.shape[0], X_test.shape[0]
            )
        )
        X_train = X_train.drop(["labels"], axis=1)
        X_test = X_test.drop(["labels"], axis=1)

        train_path = os.path.join(args.data_dir, "train.csv")
        logging.info("saving training set to {}".format(train_path))
        X_train.to_csv(train_path, index=False)

        valid_path = os.path.join(args.data_dir, "valid.csv")
        logging.info("saving validation set to {}".format(valid_path))
        X_test.to_csv(valid_path, index=False)
    else:
        train_path = os.path.join(args.data_dir, "train.csv")
        logging.info("saving training set to {}".format(train_path))
        df.to_csv(train_path, index=False)

    logging.info("dataset build finished!")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = ArgParser()
    args = parser.parse()

    logging.info("setting random seed to {}...".format(args.random_seed))
    set_random_seed(args.random_seed)

    make_dataset(args)
