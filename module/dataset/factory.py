from torch.utils.data import DataLoader

from module.dataset import MultiTaskDataset


def create_train_valid_dataset(args, mode=None):

    trainset = MultiTaskDataset(args.data_dir, mode="train", bert_config=args.bert_config)
    train_dataloader = DataLoader(
        trainset,
        batch_size=args.bach_size,
        num_workers=args.num_workers,
        collate_fn=trainset.collate_fn,
        shuffle=True,
    )

    validset = MultiTaskDataset(args.data_dir, mode="valid", bert_config=args.bert_config)
    valid_dataloader = DataLoader(
        validset,
        batch_size=args.bach_size,
        num_workers=args.num_workers,
        collate_fn=validset.collate_fn,
        shuffle=False,
    )

    if mode == "test":
        testset = MultiTaskDataset(
            args.data_dir, mode="test", bert_config=args.bert_config, test_path=args.test_data
        )
        test_dataloader = DataLoader(
            testset,
            batch_size=args.bach_size,
            num_workers=args.num_workers,
            collate_fn=testset.collate_fn,
            shuffle=False,
        )

        return {
            "trainset": trainset,
            "train_dataloader": train_dataloader,
            "validset": validset,
            "valid_dataloader": valid_dataloader,
            "testset": testset,
            "test_dataloader": test_dataloader,
        }

    return {
        "trainset": trainset,
        "train_dataloader": train_dataloader,
        "validset": validset,
        "valid_dataloader": valid_dataloader,
    }
