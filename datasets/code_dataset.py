import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.util import to_variable, try_cuda


class CodeDataset(data.Dataset):
    """
    PyTorch dataset that groups samples from an underlying dataset together so
    that they are ready for encoding.
    """

    def __init__(self, base_model, num_classes, base_dataset, base_dataset_dir,
                 ec_k, is_train, base_transform=None, code_transform=None):
        """
        Parameters
        ----------
        base_model: ``torch.nn.Module``
            Base model on which inference is being performed and over which a
            code imparts resilience.
        num_classes: int
            The number of classes in the underlying dataset.
        base_dataset: ``torchvision.datasets.Dataset``
            A dataset from the datasets provided by torchvision.
        base_dataset_dir: str
            Location where ``base_dataset`` has been or will be saved. This
            avoids re-downloading the dataset.
        ec_k: int
            Number of samples from ``base_dataset`` that will be encoded
            together.
        is_train: bool
            Whether this dataset is the training samples.
        base_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples when generating base model
            outputs that will (potentially) be used as labels.
        code_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples prior to encoding. These
            transforms may differ from those in `base_transform` as one
            may wish to include transformations such as random cropping and
            rotating of images so as to reduce overfiting. Such transformations
            would not be included in `base_transform` as they could lead to
            noisy labels being generated.
        """
        self.base_model = base_model
        self.ec_k = ec_k

        if base_transform is None:
            base_transform = transforms.ToTensor()

        # Create the datasets from the underlying `base_model_dataset`.
        # When generating outputs from running samples through the base model,
        # we do apply `base_transform`.
        self.dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                    download=True, transform=base_transform)

        # Since we are not directly calling this DataLoader when we perform
        # iterations when training a code, it is OK not to shuffle the
        # underlying dataset.
        dataloader = data.DataLoader(self.dataset, batch_size=32,
                                     shuffle=False)

        in_size = self.dataset[0][0].view(-1).size(0)
        self.num_channels = self.dataset[0][0].size(0)
        if self.num_channels > 1:
            assert self.num_channels == 3, "Only currently support 3 channels for multi-channel input"

        # Preprate data, outputs from base model, and the true labels for
        # samples. We will populate these tensors so that we can later access
        # them without pulling PIL images from the underlying dataset.
        self.data = torch.zeros(len(self.dataset), in_size)
        self.outputs = torch.zeros(len(self.dataset), num_classes)
        self.true_labels = torch.zeros(len(self.dataset), 1)

        cur = 0
        for inputs, targets in dataloader:
            inputs = to_variable(inputs.squeeze(1).view(inputs.size(0), -1))
            x = self.base_model(inputs)
            last = cur + inputs.size(0)
            self.data[cur:last, :] = inputs.data
            self.outputs[cur:last, :] = x.data
            self.true_labels[cur:last] = targets
            cur = last

        # Calculate the accuracy of the bse model with respect to this dataset.
        base_model_preds = torch.max(self.outputs, dim=1)[1]
        correct_preds = (base_model_preds ==
                         self.true_labels.squeeze(1).long())
        base_model_num_correct = torch.sum(correct_preds)
        base_model_num_tried = self.outputs.size(0)
        base_model_accuracy = base_model_num_correct / base_model_num_tried
        if is_train:
            name = "train"
        else:
            name = "validation"
        print("Base model", name, "accuracy is", base_model_num_correct, "/",
              base_model_num_tried, "=", base_model_accuracy)

        # Move data, outputs, and true labels to GPU for fast access.
        self.data = try_cuda(self.data)
        self.outputs = try_cuda(self.outputs)
        self.true_labels = try_cuda(self.true_labels.long())

        # If extra transformations are passed, create a new dataset containing
        # these so that a caller can pull new, transformed samples with calls
        # to `__getitem__`.
        if code_transform is not None:
            self.dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                        download=True, transform=code_transform)
            self.extra_transforms = True
        else:
            self.extra_transforms = False

    def __getitem__(self, idx):
        # If there are extra transformations to perform, we pull directly from
        # the underlying dataset rather than from the cached `data` tensor
        # because we'd like a new sample, and extra transformations often
        # contain some random components.
        #
        # Note, however, that even though we are pulling a "new" sample from
        # the underlying dataset, we will still use thes same output for the
        # sample as we calculated when we initially performed inference to get
        # the `outputs` tensor during `__init__`. This avoids having to perform
        # inference over the base model in-line with `__getitem__` calls.
        if self.extra_transforms:
            data, _ = self.dataset[idx]
            data = data.view(-1)
        else:
            data = self.data[idx]

        return data, self.outputs[idx], self.true_labels[idx]

    def __len__(self):
        # Number of samples in an epoch is equal to the number of `ec_k`-sized
        # groups are contained in our dataset.
        return (self.data.size(0) // self.ec_k) * self.ec_k

    def encoder_in_dim(self):
        """
        Returns dimensionality of input that will be given to the encoder.
        """
        return self.data.size(1) // self.num_channels

    def decoder_in_dim(self):
        """
        Returns dimensionality of input that will be given to the decoder.
        """
        return self.outputs.size(1)


class MNISTCodeDataset(CodeDataset):
    def __init__(self, base_model, ec_k, is_train):
        base_dataset = datasets.MNIST
        base_dataset_dir = "data/mnist"
        super().__init__(base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, is_train=is_train, num_classes=10)


class FashionMNISTCodeDataset(CodeDataset):
    def __init__(self, base_model, ec_k, is_train):
        base_dataset = datasets.FashionMNIST
        base_dataset_dir = "data/fashion-mnist"
        super().__init__(base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, is_train=is_train, num_classes=10)


class CIFAR10CodeDataset(CodeDataset):
    def __init__(self, base_model, ec_k, is_train):
        base_dataset = datasets.CIFAR10
        base_dataset_dir = "data/cifar10"

        # For `base_transform`, we only apply normalization.
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])

        # We add extra transformations for CIFAR-10 as is done in:
        #   https://github.com/kuangliu/pytorch-cifar
        code_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])

        super().__init__(base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, is_train=is_train, num_classes=10,
                         base_transform=base_transform,
                         code_transform=code_transform)
