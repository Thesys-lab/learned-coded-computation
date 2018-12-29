# Configuration Files
This repository uses JSON-formatted configuration files to describe parameters
used for training. In the following, we walk through all parameters that must
be set in a configuration file by using [mnist/base-mlp/k2/kl-base/mlp/conf.json](mnist/base-mlp/k2/kl-base/mlp/conf.json)
as an example.

Many configuration arguments use the following format:
```json
"Name": {
  "args": {
    "arg0": 3,
    "arg1": "hello"
  },
  "class": "path.to.class.ClassName"
}
```
These specify a particular class that should be constructed. The name of the class, proceeded
by the import path relative to the root of this repository, is included under "class". The
arguments to the class that are modified by user-level configurations are listed under "args".
Note that not all arguments to a class's constructor may be included in the configuration file.
Only those arguments which are tunable for training are included in configuration files -- others
are supplied by the underlying training script. As a general rule of thumb, all arguments that are
included in one of the example configuration files in this directory should be included when
developing your own configuration file.

### Base Model
```
"BaseModel": {
  "class": "base_models.base_mlp.BaseMLP"
}
```
* `class`: Path to the base model architecture.

### Decoder
```json
"Decoder": {
  "class": "coders.mlp_coder.MLPDecoder"
}
```
* `class`: Path to the decoding function architecture.

### Decoder Optimizer
```json
"DecoderOptimizer": {
  "args": {
    "lr": 0.001,
    "weight_decay": 1e-05
  },
  "class": "torch.optim.Adam"
}
```
This class room for leeway -- you should be able to use most optimizers from the `torch.optim` family.
* `class`: PyTorch optimizer class.
* `args`: Parameters to the optimizer, as defined by PyTorch documentation.

### Encoder
```json
"Encoder": {
  "class": "coders.mlp_coder.MLPEncoder"
}
```
* `class`: Path to the encoding function architecture. Current options are `coders.mlp_coder.MLPEncoder`
  and `coders.conv_encoder.ConvEncoder`.

### Encoder Optimizer
Same format as "Decoder Optimizer" above.

### Loss Function
```json
"Loss": {
  "class": "loss.kldiv.KLDivLoss"
}
```
* `class`: Class used for calculating loss. Options used for configurations in the paper are
  `loss.kldiv.KLDivLoss` (KL-Base in paper), `torch.nn.MSELoss` (MSE-Base in paper), and
  `torch.nn.CrossEntropyLoss` (XENT-Label in paper). `loss.kldiv.KLDivLoss` implements a
  wrapper around `torch.nn.KLDivLoss` that formats inputs appropriately.

### Datasets
```json
"Dataset": {
  "class": "datasets.code_dataset.MNISTCodeDataset"
}
```
* `class`: Path to dataset class. Currently, only those implemented in the [datasets](../datasets)
  directory will work.

### Base model details
```json
"base_model_file": "base_model_trained_files/mnist/base-mlp/model.t7",
"base_model_input_size": [-1, 784]
```
* `base_model_file`: Path to the pre-trained base model parameters. All those
  for configurations included in the paper's experiments are located in
  [base_model_trained_files](../base_model_trained_files).
* `base_model_input_size`: Dimensions of inputs to the base model. The first
  entry in the list should always be -1 to specify arbitrary batch size. This
  example shows that, for BaseMLP, input images from the MNIST dataset should
  be flattened from 28x28 form to a vector of length 784. For multi-channel,
  convolutional base models, like ResNet-18 used for CIFAR-10, this value might
  be `[-1, 3, 32, 32]`, as depcited in [cifar10/resnet18/k2/kl-base/mlp/conf.json](cifar10/resnet18/k2/kl-base/mlp/conf.json).

### Other training flags
```json
"batch_size": 64,
"ec_k": 2,
"ec_r": 1,
"final_epoch": 500,
"save_dir": "save/mnist/base-mlp/k2/kl-base/mlp"
```
* `batch_size`: Minibatch size used in training. This specifies the number of
  samples that will be generated from the encoder on each minibatch. That is,
  in actuality, `batch_size * ec_k` samples will be drawn from the underlying
  dataset.
* `ec_k`: Value of parameter k, that is, the number of images from the underlying
  dataset that will be encoded together.
* `ec_r`: Value of parameter r, that is, how many parity images are generated from
  the encoding of `ec_k` images. Currently this should always remain 1. The
  repository currently does not support other values of `ec_r`.
* `final_epoch`: The final epoch number of training.
* `save_dir`: Directory to which results should be saved. The results saved to
  a particular file are detailed in [save/README.md](../save/README.md).

### Continuing training from a checkpoint
To continue training from a checkpoint file that as been saved during a previous
training run, add the following flag:
```json
"continue_from_file": "path/to/current.pth"
```
where "current.pth" is the checkpoint to start from.
