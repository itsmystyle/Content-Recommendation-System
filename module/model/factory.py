from module.model import BertContentBased


def create_model(args, num_class_1, num_class_2):
    return BertContentBased.from_pretrained(
        args.bert_config,
        num_class_1=num_class_1,
        num_class_2=num_class_2,
        compress_dim=args.feature_dim,
    )
