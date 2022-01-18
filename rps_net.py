"""
RPS network script with resnet-18
Copyright (c) Jathushan Rajasegaran, 2019
"""

from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RPS_net(nn.Module):
    def __init__(self, args):
        super(RPS_net, self).__init__()
        self.args = args
        self.final_layers = []
        self.init(None)

    def init(self, best_path):

        """Initialize all parameters"""
        self.conv1 = []
        self.conv2 = []
        self.conv3 = []
        self.conv4 = []
        self.conv5 = []
        self.conv6 = []
        self.conv7 = []
        self.conv8 = []
        self.conv9 = []
        self.fc1 = []

        # conv1
        for i in range(self.args.M):
            exec(
                "self.m1"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),nn.BatchNorm2d(64),nn.ReLU())"
            )
            exec("self.conv1.append(self.m1" + str(i) + ")")

        # conv2
        for i in range(self.args.M):
            exec(
                "self.m2"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))"
            )
            exec("self.conv2.append(self.m2" + str(i) + ")")

        # conv3
        for i in range(self.args.M):
            exec(
                "self.m3"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64))"
            )
            exec("self.conv3.append(self.m3" + str(i) + ")")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4
        for i in range(self.args.M):
            exec(
                "self.m4"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))"
            )
            exec("self.conv4.append(self.m4" + str(i) + ")")
        exec(
            "self.m4"
            + str("x")
            + " = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU())"
        )
        exec("self.conv4.append(self.m4" + str("x") + ")")

        # conv5
        for i in range(self.args.M):
            exec(
                "self.m5"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128))"
            )
            exec("self.conv5.append(self.m5" + str(i) + ")")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv6
        for i in range(self.args.M):
            exec(
                "self.m6"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))"
            )
            exec("self.conv6.append(self.m6" + str(i) + ")")
        exec(
            "self.m6"
            + str("x")
            + " = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU())"
        )
        exec("self.conv6.append(self.m6" + str("x") + ")")

        # conv7
        for i in range(self.args.M):
            exec(
                "self.m7"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256))"
            )
            exec("self.conv7.append(self.m7" + str(i) + ")")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv8
        for i in range(self.args.M):
            exec(
                "self.m8"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))"
            )
            exec("self.conv8.append(self.m8" + str(i) + ")")
        exec(
            "self.m8"
            + str("x")
            + " = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())"
        )
        exec("self.conv8.append(self.m8" + str("x") + ")")

        # conv9
        for i in range(self.args.M):
            exec(
                "self.m9"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512))"
            )
            exec("self.conv9.append(self.m9" + str(i) + ")")

        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(512, 1000)
            self.final_layers.append(self.final_layer1)

        self.cuda()

    def forward(self, x, path, last):

        M = self.args.M
        div = 1
        p = 0.5

        y = self.conv1[0](x)
        for j in range(1, self.args.M):
            if path[0][j] == 1:
                y += self.conv1[j](x)
        x = F.relu(y)

        y = self.conv2[0](x)
        for j in range(1, self.args.M):
            if path[1][j] == 1:
                y += self.conv2[j](x)
        x = y + x
        x = F.relu(x)

        y = self.conv3[0](x)
        for j in range(1, self.args.M):
            if path[2][j] == 1:
                y += self.conv3[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool1(x)

        y = self.conv4[-1](x)
        for j in range(self.args.M):
            if path[3][j] == 1:
                y += self.conv4[j](x)
        x = y
        x = F.relu(x)

        y = self.conv5[0](x)
        for j in range(1, self.args.M):
            if path[4][j] == 1:
                y += self.conv5[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool2(x)

        y = self.conv6[-1](x)
        for j in range(self.args.M):
            if path[5][j] == 1:
                y += self.conv6[j](x)
        x = y
        x = F.relu(x)

        y = self.conv7[0](x)
        for j in range(1, self.args.M):
            if path[6][j] == 1:
                y += self.conv7[j](x)
        x = y
        x = F.relu(x)
        x = self.pool3(x)

        y = self.conv8[-1](x)
        for j in range(self.args.M):
            if path[7][j] == 1:
                y += self.conv8[j](x)
        x = y
        x = F.relu(x)

        y = self.conv9[0](x)
        for j in range(1, self.args.M):
            if path[8][j] == 1:
                y += self.conv9[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool4(x)

        x = F.avg_pool2d(x, (7, 7), stride=(1, 1))
        x = x.view(-1, 512)
        x = self.final_layers[last](x)

        return x


class RPS_net_cifar(nn.Module):
    def __init__(self, args):
        super(RPS_net_cifar, self).__init__()
        self.args = args
        self.final_layers = []
        self.init(None)

    def init(self, best_path):

        """Initialize all parameters"""
        self.conv1 = []
        self.conv2 = []
        self.conv3 = []
        self.conv4 = []
        self.conv5 = []
        self.conv6 = []
        self.conv7 = []
        self.conv8 = []
        self.conv9 = []
        self.fc1 = []

        div = 1
        a1 = 64 // div

        a2 = 64 // div
        a3 = 128 // div
        a4 = 256 // div
        a5 = 512 // div

        self.a5 = a5
        # conv1
        for i in range(self.args.M):
            exec(
                "self.m1"
                + str(i)
                + " = nn.Sequential(nn.Conv2d(3, "
                + str(a1)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a1)
                + "),nn.ReLU())"
            )
            exec("self.conv1.append(self.m1" + str(i) + ")")

        # conv2
        for i in range(self.args.M):
            exec(
                "self.m2"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a1)
                + ", "
                + str(a2)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a2)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a2)
                + ", "
                + str(a2)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a1)
                + "))"
            )
            exec("self.conv2.append(self.m2" + str(i) + ")")

        # conv3
        for i in range(self.args.M):
            exec(
                "self.m3"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a2)
                + ", "
                + str(a2)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a2)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a2)
                + ", "
                + str(a2)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a2)
                + "))"
            )
            exec("self.conv3.append(self.m3" + str(i) + ")")

        # conv4
        for i in range(self.args.M):
            exec(
                "self.m4"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a2)
                + ", "
                + str(a3)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a3)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a3)
                + ", "
                + str(a3)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a3)
                + "))"
            )
            exec("self.conv4.append(self.m4" + str(i) + ")")
        exec(
            "self.m4"
            + str("x")
            + " = nn.Sequential(nn.Conv2d("
            + str(a2)
            + ", "
            + str(a3)
            + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
            + str(a3)
            + "),nn.ReLU())"
        )
        exec("self.conv4.append(self.m4" + str("x") + ")")

        # conv5
        for i in range(self.args.M):
            exec(
                "self.m5"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a3)
                + ", "
                + str(a3)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a3)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a3)
                + ", "
                + str(a3)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a3)
                + "))"
            )
            exec("self.conv5.append(self.m5" + str(i) + ")")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv6
        for i in range(self.args.M):
            exec(
                "self.m6"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a3)
                + ", "
                + str(a4)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a4)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a4)
                + ", "
                + str(a4)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a4)
                + "))"
            )
            exec("self.conv6.append(self.m6" + str(i) + ")")
        exec(
            "self.m6"
            + str("x")
            + " = nn.Sequential(nn.Conv2d("
            + str(a3)
            + ", "
            + str(a4)
            + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
            + str(a4)
            + "),nn.ReLU())"
        )
        exec("self.conv6.append(self.m6" + str("x") + ")")

        # conv7
        for i in range(self.args.M):
            exec(
                "self.m7"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a4)
                + ", "
                + str(a4)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a4)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a4)
                + ", "
                + str(a4)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a4)
                + "))"
            )
            exec("self.conv7.append(self.m7" + str(i) + ")")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv8
        for i in range(self.args.M):
            exec(
                "self.m8"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a4)
                + ", "
                + str(a5)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a5)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a5)
                + ", "
                + str(a5)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a5)
                + "))"
            )
            exec("self.conv8.append(self.m8" + str(i) + ")")
        exec(
            "self.m8"
            + str("x")
            + " = nn.Sequential(nn.Conv2d("
            + str(a4)
            + ", "
            + str(a5)
            + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
            + str(a5)
            + "),nn.ReLU())"
        )
        exec("self.conv8.append(self.m8" + str("x") + ")")

        # conv9
        for i in range(self.args.M):
            exec(
                "self.m9"
                + str(i)
                + " = nn.Sequential(nn.Conv2d("
                + str(a5)
                + ", "
                + str(a5)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a5)
                + "),nn.ReLU(), nn.Conv2d("
                + str(a5)
                + ", "
                + str(a5)
                + ", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("
                + str(a5)
                + "))"
            )
            exec("self.conv9.append(self.m9" + str(i) + ")")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(a5, 100)
            self.final_layers.append(self.final_layer1)

        self.cuda()

    def forward(self, x, path, last):

        M = self.args.M
        div = 1
        p = 0.5

        y = self.conv1[0](x)
        for j in range(1, self.args.M):
            if path[0][j] == 1:
                y += self.conv1[j](x)
        x = F.relu(y)

        y = self.conv2[0](x)
        for j in range(1, self.args.M):
            if path[1][j] == 1:
                y += self.conv2[j](x)
        x = y + x
        x = F.relu(x)

        y = self.conv3[0](x)
        for j in range(1, self.args.M):
            if path[2][j] == 1:
                y += self.conv3[j](x)
        x = y + x
        x = F.relu(x)

        y = self.conv4[-1](x)
        for j in range(self.args.M):
            if path[3][j] == 1:
                y += self.conv4[j](x)
        x = y
        x = F.relu(x)

        y = self.conv5[0](x)
        for j in range(1, self.args.M):
            if path[4][j] == 1:
                y += self.conv5[j](x)
        x = y + x
        x = F.relu(x)
        x = self.pool1(x)

        y = self.conv6[-1](x)
        for j in range(self.args.M):
            if path[5][j] == 1:
                y += self.conv6[j](x)
        x = y
        x = F.relu(x)

        y = self.conv7[0](x)
        for j in range(1, self.args.M):
            if path[6][j] == 1:
                y += self.conv7[j](x)
        x = y
        x = F.relu(x)
        x = self.pool2(x)

        y = self.conv8[-1](x)
        for j in range(self.args.M):
            if path[7][j] == 1:
                y += self.conv8[j](x)
        x = y
        x = F.relu(x)

        y = self.conv9[0](x)
        for j in range(1, self.args.M):
            if path[8][j] == 1:
                y += self.conv9[j](x)
        x = y + x
        x = F.relu(x)
        #             x = self.pool3(x)

        x = F.avg_pool2d(x, (8, 8), stride=(1, 1))
        x = x.view(-1, self.a5)
        x = self.final_layers[last](x)

        return x


class RPS_net_mlp(nn.Module):
    def __init__(self, M: int):
        super(RPS_net_mlp, self).__init__()
        self.final_layers = []

        # Initialize all parameters
        self.M = M
        self.mlp1 = []
        self.mlp2 = []
        self.fc1 = []

        # mlp1
        for i in range(self.M):
            newModuleName = f"m1{i}"
            setattr(
                self, newModuleName, nn.Linear(32 * 32, 400)
            )  # add module to RPS_net
            self.mlp1.append(getattr(self, newModuleName))  # add module to mlp1

        # mlp2
        for i in range(self.M):
            newModuleName = f"m2{i}"
            setattr(self, newModuleName, nn.Linear(400, 128))  # add module to RPS_net
            self.mlp2.append(getattr(self, newModuleName))  # add module to mlp2

        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(128, 10)
            self.final_layers.append(self.final_layer1)

        self.pathLayers = [self.mlp1, self.mlp2]

        self.cuda()

    def __str__(self) -> str:
        return f"    Total params: {(sum(p.numel() for p in self.parameters()) / 1000000.0):.2f}M"

    def parameters(
        self, trainablePath: np.array = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Get training parameters of modules to train, given a path

        Args:
            trainablePath (numpy.array):

        Returns:
            List[Dict[torch.Tensor]]: [description]
        """
        if trainablePath is None:
            return super().parameters(True)
        else:

            trainableParams = []
            for layerIdx, layer in enumerate(self.pathLayers):
                for moduleIdx in range(self.M):
                    if trainablePath[layerIdx][moduleIdx]:
                        trainableParams.append(
                            {"params": layer[moduleIdx].parameters()}
                        )
                    else:
                        layer[layerIdx].requires_grad = False
            trainableParams.append({"params": self.final_layers[-1].parameters()})
        return trainableParams

    def forward(self, x, path, last):
        y = self.mlp1[0](x)
        for j in range(1, self.M):
            if path[0][j]:  # infer only on activated modules
                y += self.mlp1[j](x)  # sum on activated modules
        x = F.relu(y)

        y = self.mlp2[0](x)
        for j in range(1, self.M):
            if path[1][j]:
                y += self.mlp2[j](x)
        x = F.relu(y)

        x = self.final_layers[last](x)

        return x
