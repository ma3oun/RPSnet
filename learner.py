import os
import torch
from utils import Logger, AverageMeter, accuracy, savefig
import torch.optim as optim
import time
import torch.nn as nn
from tqdm import tqdm


class Learner:
    def __init__(
        self,
        model,
        args,
        trainloader,
        testloader,
        old_model,
        use_cuda,
        path,
        fixed_path,
        train_path,
        infer_path,
        title,
    ):
        self.model = model
        self.args = args
        self.title = f"{title}-{self.args.arch}"
        self.trainloader = trainloader
        self.use_cuda = use_cuda
        self.state = {
            key: value
            for key, value in self.args.__dict__.items()
            if not key.startswith("__") and not callable(key)
        }
        self.best_acc = 0
        self.testloader = testloader
        self.start_epoch = self.args.start_epoch
        self.test_loss = 0.0
        self.path = path
        self.fixed_path = fixed_path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc = 0.0
        self.train_loss, self.train_acc = 0.0, 0.0
        self.old_model = old_model
        if self.args.sess > 0:
            self.old_model.eval()

        trainable_params = self.model.parameters(self.train_path)
        print("Number of layers being trained : ", len(trainable_params))

        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

    def learn(self):
        if self.args.resume:
            # Load checkpoint.
            print("==> Resuming from checkpoint..")
            assert os.path.isfile(
                self.args.resume
            ), "Error: no checkpoint directory found!"
            self.args.checkpoint = os.path.dirname(self.args.resume)
            checkpoint = torch.load(self.args.resume)
            self.best_acc = checkpoint["best_acc"]
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger = Logger(
                os.path.join(self.args.checkpoint, "log.txt"),
                title=self.title,
                resume=True,
            )
        else:
            logger = Logger(
                os.path.join(
                    self.args.checkpoint,
                    f"session_{self.args.sess}_{self.args.test_case}_log.txt",
                ),
                title=self.title,
            )
            logger.set_names(
                [
                    "Learning Rate",
                    "Train Loss",
                    "Valid Loss",
                    "Train Acc.",
                    "Valid Acc.",
                    "Best Acc",
                ]
            )
        if self.args.evaluate:
            print("\nEvaluation only")
            self.test(self.infer_path)
            print(f" Test Loss: {self.test_loss:.8f}, Test Acc:  {self.test_acc:.2f}")
            return

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(epoch)

            print(
                f"\nEpoch: [{epoch + 1} | {self.args.epochs}] LR: {self.state['lr']} Sess: {self.args.sess}"
            )

            self.train(self.infer_path)
            self.test(self.infer_path)

            # append logger file
            logger.append(
                [
                    self.state["lr"],
                    self.train_loss,
                    self.test_loss,
                    self.train_acc,
                    self.test_acc,
                    self.best_acc,
                ]
            )

            # save model
            is_best = self.test_acc > self.best_acc
            self.best_acc = max(self.test_acc, self.best_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "acc": self.test_acc,
                    "best_acc": self.best_acc,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best,
                checkpoint=self.args.savepoint,
                filename=f"session_{self.args.sess}_{self.args.test_case}_checkpoint.pth.tar",
                session=self.args.sess,
                test_case=self.args.test_case,
            )

        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, "log.eps"))

        print("Best acc:")
        print(self.best_acc)

    def train(self, path):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        lossFn = nn.CrossEntropyLoss(reduction="batchmean")
        distillLossFn = nn.KLDivLoss(reduction="batchmean")

        with tqdm(
            enumerate(self.trainloader),
            total=len(self.trainloader),
            desc="Train: ",
        ) as bar:
            for batch_idx, (inputs, targets) in bar:
                # measure data loading time
                data_time.update(time.time() - end)

                if self.use_cuda:
                    inputs, targets = (
                        inputs.cuda(),
                        targets.cuda(),
                    )

                # compute output
                outputs = self.model(inputs, path).squeeze()
                loss = lossFn(outputs, targets)
                loss_dist = 0

                ## distillation loss
                if self.args.sess > 0:
                    outputs_old = self.old_model(inputs, path).squeeze()

                    if self.args.sess in range(1 + self.args.jump):
                        cx = 1
                    else:
                        cx = self.args.rigidness_coff * (
                            self.args.sess - self.args.jump
                        )

                    loss_dist = cx * distillLossFn(outputs, outputs_old).clamp(min=0.0)

                loss += loss_dist

                # measure accuracy and record loss
                if self.args.dataset == "MNIST":
                    prec1, prec5 = accuracy(
                        output=outputs.data[
                            :, 0 : self.args.class_per_task * (1 + self.args.sess)
                        ],
                        target=targets.cuda().data,
                        topk=(1, 1),
                    )
                else:
                    prec1, prec5 = accuracy(
                        output=outputs.data[
                            :, 0 : self.args.class_per_task * (1 + self.args.sess)
                        ],
                        target=targets.data,
                        topk=(1, 5),
                    )
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.set_description(
                    f"Train: Loss: {losses.avg:.4f} | Dist: {loss_dist:.4f} | top1: {top1.avg: .4f} | top5: {top5.avg: .4f}"
                )

        self.train_loss, self.train_acc = losses.avg, top1.avg

    def test(self, path):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        lossFn = nn.CrossEntropyLoss()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with tqdm(
            enumerate(self.testloader),
            total=len(self.testloader),
            desc="Test: ",
        ) as bar:
            for batch_idx, (inputs, targets) in bar:
                # measure data loading time
                data_time.update(time.time() - end)

                if self.use_cuda:
                    inputs, targets = (
                        inputs.cuda(),
                        targets.cuda(),
                    )

                outputs = self.model(inputs, path).squeeze()

                loss = lossFn(outputs, targets)

                # measure accuracy and record loss
                if self.args.dataset == "MNIST":
                    prec1, prec5 = accuracy(
                        outputs.data[
                            :, 0 : self.args.class_per_task * (1 + self.args.sess)
                        ],
                        targets.cuda().data,
                        topk=(1, 1),
                    )
                else:
                    prec1, prec5 = accuracy(
                        outputs.data[
                            :, 0 : self.args.class_per_task * (1 + self.args.sess)
                        ],
                        targets.cuda().data,
                        topk=(1, 5),
                    )

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.set_description(
                    f"Test: Loss: {losses.avg:.4f} | top1: {top1.avg: .4f} | top5: {top5.avg: .4f}"
                )

        self.test_loss = losses.avg
        self.test_acc = top1.avg

    def save_checkpoint(
        self,
        state,
        is_best,
        checkpoint="checkpoint",
        filename="checkpoint.pth.tar",
        session=0,
        test_case=0,
    ):
        if is_best:
            torch.save(
                state,
                os.path.join(
                    checkpoint, f"session_{session}_{test_case}_model_best.pth.tar"
                ),
            )

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state["lr"] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.state["lr"]

    def get_confusion_matrix(self, path):

        confusion_matrix = torch.zeros(100, 100)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                outputs = self.model(inputs, path)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix
