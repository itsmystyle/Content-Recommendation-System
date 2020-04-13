import os
import logging

import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optim,
        criterions,
        metric,
        scheduler,
        train_dl,
        val_dl,
        writer,
        save_dir,
        device,
    ):

        self.device = device

        # Models
        self.model = model

        # Optimizer
        self.optim = optim
        self.scheduler = scheduler

        # Criterions
        self.criterions = criterions

        # Dataloader
        self.train_dataloader = train_dl
        self.valid_dataloader = val_dl

        # Utils
        self.writer = writer
        self.save_dir = save_dir
        self.metric = metric

    def fit(self, epochs):
        # train model
        logging.info("===> start training ...")
        iters = 0
        val_iters = 0
        best_acc = 0
        best_loss = 0.0

        for epoch in range(1, epochs + 1):
            iters = self._run_one_epoch(epoch, iters)

            val_iters, best_acc, best_loss = self._eval_one_epoch(
                epoch, val_iters, best_acc, best_loss
            )

        logging.info(
            "===> training ended, Best model Accuracy: {} with loss of {}".format(
                best_acc, best_loss
            )
        )

    def _run_one_epoch(self, epoch, iters):
        # train model one epoch
        self.model.train()

        trange = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Epoch {}".format(epoch),
        )

        Loss = {"class_1": [], "class_2": [], "total": []}
        for metric in self.metric.values():
            metric.reset()

        for idx, batch in trange:

            self.optim.zero_grad()

            items = self._run_one_iter(batch)

            # update metric
            self.metric["class_1"].update(
                items["class_1_logits"].detach().cpu().numpy(), batch["class_1"].numpy()
            )
            self.metric["class_2"].update(
                items["class_2_logits"].detach().cpu().numpy(), batch["class_2"].numpy()
            )

            loss = items["p_loss"] + items["c_loss"]
            loss.backward()
            self.optim.step()
            if self.scheduler is not None:
                self.scheduler.step()

            Loss["class_1"].append(items["p_loss"].item())
            Loss["class_2"].append(items["c_loss"].item())
            Loss["total"].append(loss.item() / 2)

            # update tqdm
            postfix_dict = {
                "P_Loss": "{:.5f}".format(np.array(Loss["class_1"]).mean()),
                "P_Acc": "{:.5f}".format(self.metric["class_1"].get_score()),
                "C_Loss": "{:.5f}".format(np.array(Loss["class_2"]).mean()),
                "C_Acc": "{:.5f}".format(self.metric["class_2"].get_score()),
                "Avg.": "{:.5f}".format(np.array(Loss["total"]).mean()),
            }
            trange.set_postfix(**postfix_dict)

            # writer write loss per iteration
            self.writer.add_scalars(
                "train_loss",
                {
                    "class_1": np.array(Loss["class_1"]).mean(),
                    "class_2": np.array(Loss["class_2"]).mean(),
                    "total": np.array(Loss["total"]).mean(),
                },
                iters,
            )
            self.writer.add_scalars(
                "train_accuracy",
                {
                    "class_1": self.metric["class_1"].get_score(),
                    "class_2": self.metric["class_2"].get_score(),
                },
                iters,
            )

            iters += 1

        # writer write loss per epoch
        self.writer.add_scalars(
            "train_epoch",
            {
                "loss": np.array(Loss["total"]).mean(),
                "accuracy": (
                    self.metric["class_1"].get_score() + self.metric["class_2"].get_score()
                )
                / 2,
            },
            epoch,
        )

        return iters

    def _eval_one_epoch(self, epoch, val_iters, best_acc, best_loss):
        # evaluate model one epoch
        self.model.eval()

        Loss = {"class_1": [], "class_2": [], "total": []}
        for metric in self.metric.values():
            metric.reset()

        trange = tqdm(
            enumerate(self.valid_dataloader), total=len(self.valid_dataloader), desc="Valid",
        )

        with torch.no_grad():
            for idx, batch in trange:
                # prepare training data
                items = self._run_one_iter(batch)

                # update metric
                self.metric["class_1"].update(
                    items["class_1_logits"].detach().cpu().numpy(), batch["class_1"].numpy()
                )
                self.metric["class_2"].update(
                    items["class_2_logits"].detach().cpu().numpy(), batch["class_2"].numpy()
                )

                loss = items["p_loss"].item() + items["c_loss"].item()
                Loss["class_1"].append(items["p_loss"].item())
                Loss["class_2"].append(items["c_loss"].item())
                Loss["total"].append(loss / 2)

                # update tqdm
                postfix_dict = {
                    "P_Loss": "{:.5f}".format(np.array(Loss["class_1"]).mean()),
                    "P_Acc": "{:.5f}".format(self.metric["class_1"].get_score()),
                    "C_Loss": "{:.5f}".format(np.array(Loss["class_2"]).mean()),
                    "C_Acc": "{:.5f}".format(self.metric["class_2"].get_score()),
                    "Avg.": "{:.5f}".format(np.array(Loss["total"]).mean()),
                }
                trange.set_postfix(**postfix_dict)

                # writer write loss per iteration
                self.writer.add_scalars(
                    "val_loss",
                    {
                        "class_1": np.array(Loss["class_1"]).mean(),
                        "class_2": np.array(Loss["class_2"]).mean(),
                        "Avg.": np.array(Loss["total"]).mean(),
                    },
                    val_iters,
                )
                self.writer.add_scalars(
                    "val_accuracy",
                    {
                        "class_1": self.metric["class_1"].get_score(),
                        "class_2": self.metric["class_2"].get_score(),
                    },
                    val_iters,
                )

                val_iters += 1

            current_acc = (
                self.metric["class_1"].get_score() + self.metric["class_2"].get_score()
            ) / 2

            # writer write loss per epoch
            self.writer.add_scalars(
                "val_epoch",
                {"loss": np.array(Loss["total"]).mean(), "accuracy": current_acc},
                epoch,
            )

            # save best acc model
            if current_acc > best_acc:
                print("Best model saved!")
                best_acc = current_acc
                best_loss = np.array(Loss["total"]).mean()
                self.save(
                    os.path.join(
                        self.save_dir,
                        # "model_best_{:.5f}_{:.5f}.pth.tar".format(best_acc, np.array(Loss).mean())
                        "model_best.pth.tar",
                    )
                )

        return val_iters, best_acc, best_loss

    def _run_one_iter(self, batch):
        # prepare training data
        batch = {
            k: v.to(self.device)
            for k, v in batch.items()
            if v is not None and k != "id" and not isinstance(v, list)
        }

        # calculate label loss
        preds = self.model(batch)

        # calculate loss
        p_loss = self.calc_loss(preds["class_1"], preds["features"], batch["class_1"], "class_1")
        c_loss = self.calc_loss(
            preds["class_2"], preds["features"], batch["class_2"], "class_2"
        )

        return {
            "p_loss": p_loss,
            "c_loss": c_loss,
            "features": preds["features"],
            "class_1_logits": preds["class_1"],
            "class_2_logits": preds["class_2"],
        }

    def calc_loss(self, preds, features, labels, loss_name):
        # calculate loss
        loss = 0.0
        for criterion in self.criterions[loss_name]:
            name = criterion[0]
            scale = criterion[2]
            criterion = criterion[1]

            if name in ["TripletLoss", "CenterLoss"]:
                loss += scale * criterion(features, labels)
            elif name in ["ArcFaceLoss"]:
                logits = criterion[0](features, labels)
                loss += scale * criterion[1](logits, labels)
            else:
                loss += scale * criterion(preds, labels)

        return loss

    def save(self, path):
        torch.save(
            self.model.state_dict(), path,
        )
