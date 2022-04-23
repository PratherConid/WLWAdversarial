from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)

# Create data loaders.
train_dataloader = lambda batch_size: DataLoader(training_data, batch_size=batch_size)
test_dataloader = lambda batch_size: DataLoader(test_data, batch_size=batch_size)