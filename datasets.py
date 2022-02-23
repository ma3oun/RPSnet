import torch
import torchvision.transforms as transforms
from cl_datasets import Cl_dataset


class FlattenTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.flatten(img)


mnistTrainTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        FlattenTransform(),
    ]
)

mnistTestTransforms = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        FlattenTransform(),
    ]
)

mnistDataset = Cl_dataset("mnist", 10, 5, mnistTrainTransforms, mnistTestTransforms)

svhnDataset = Cl_dataset("svhn", 10, 5, mnistTrainTransforms, mnistTestTransforms)

cifarTrainTransforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifarTestTransforms = testTransforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar100Dataset = Cl_dataset(
    "cifar100", 100, 10, cifarTrainTransforms, cifarTestTransforms
)


cifar10Dataset = Cl_dataset(
    "cifar10", 10, 1, cifarTrainTransforms, cifarTestTransforms
)

