import logging

import torch.nn as nn
from module.loss import CrossEntropyLabelSmooth, ArcMarginProduct, CenterLoss, TripletLoss


def create_criterions(args, num_classes, device=None):
    criterion = []

    # Cross Entropy Loss
    if args.smoothing:
        CELoss = CrossEntropyLabelSmooth(num_classes, device=device)
    else:
        CELoss = nn.NLLLoss()

    # Arcface: Arc Large Margin Product loss
    # speacial case when using arcface, need to append CELoss to it.
    if args.arcface:
        criterion += [
            (
                "ArcFaceLoss",
                (
                    ArcMarginProduct(
                        args.feature_dim,
                        num_classes,
                        False,
                        s=args.scale,
                        m=args.margin,
                        device=device,
                    ),
                    CELoss,
                ),
                1.0,
            )
        ]
    else:
        criterion += [("CELoss", CELoss, 1.0)]

    # Triple loss
    if args.triplet:
        criterion += [("TripletLoss", TripletLoss(0.3, args.dist), 4.0)]

    # Center loss
    if args.center:
        criterion += [
            (
                "CenterLoss",
                CenterLoss(num_classes=num_classes, feat_dim=args.feature_dim, device=device,),
                1.0,
            )
        ]

    logging.info("criterions: {}".format(criterion))

    return criterion
