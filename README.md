# learned-cc
This repository contains the code associated with the paper
["Learning a Code: Machine Learning for Approximate Non-Linear
Coded-Computation"](https://arxiv.org/abs/1806.01259).

### Software requirements
This repository was developed using the following versions of software and has
not been tested using other versions.
* Python 3.6.5
* PyTorch version 0.3.0 with torchvision (will likely work with higher versions, but they have not been tested)

### Repository structure
* [base_models](base_models): Implementations of the Base-MLP and ResNet-18 base models used
  in the paper.
* [base_model_trained_files](base_model_trained_files): PyTorch model state dictionaries containing
  trained parameters for the base models used in the paper.
* [coders](coders): Implementations of MLPEncoder, ConvEncoder, and the MLP
  decoding function described in the paper.
* [conf](conf): JSON configuration files for running all experiments discussed in the paper.
  See the README in `conf` for details of the configuration file format.
* [datasets](datasets): PyTorch `Dataset` implementations for generating samples for training
  the encoding and decoding functions.
* [loss](loss): Wrappers around loss functions used for experiments.
* [save](save): Directory to which results of training are saved, when using the
  configuration files provided in [conf](conf).
* [util](util): Utility methods used throughout the repository.
* [code_trainer.py](code_trainer.py): Top-level class for managing the training of encoding and
  decoding functions over a particular base model.
* [train.py](train.py): Script for launching a training job given a configuration file.

### Quickstart
To begin training of a given configuration file, run the following command:
```python
python3 train.py conf/mnist/base-mlp/k2/kl-base/mlp/conf.json
```

Training automatically uses GPU if available. Accuracies, losses, and checkpoints
of trained parameters are saved to the location specified by `save_dir` in the
given configuration file. In all configurations provided in the [conf](conf) directory,
this defaults to a location in the directory [save](save) corresponding to the directory
hierarchy in the [conf](conf) directory.

### Adding a new configuration file
See the details in the [conf](conf) directory about the structure of
configuration files.

### Adding a new base model
See details in the [base_models](base_models) directory regarding
adding a base model that is not included in this repository.

### Adding a new dataset
See details in the [datasets](datasets) directory regarding training a learned
code using a dataset that is not included in this repository.

### Adding new encoders/decoders
See details in the [coders](coders) directory regarding adding a new
encoder/decoder architecture.

### Known issues
Two configurations occasionally (it seems due to randomness in initialization
or data sampling) fail to achieve loss better than random guessing, or plateau
at high loss. The configurations are using the `MNIST` dataset with `XENT-Label`
as loss, `k=5`, `ConvEncoder`, and for base models `Base-MLP` and `ResNet-18`.
We have found that simply restarting training early on when this occurs often
leads to this being resolved.

### Contact
Please raise an issue in this repository if you have any questions, or requests
related to this code. Other comments may be directed to Jack Kosaian ([jkosaian@cs.cmu.edu](mailto:jkosaian@cs.cmu.edu)),
but GitHub issues are preferred over email.

### Citation
```
@article{kosaian2018learning,
  title={{Learning a Code: Machine Learning for Approximate Non-Linear Coded Computation}},
  author={Kosaian, Jack and Rashmi, KV and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:1806.01259},
  year={2018}
}
```

### License
```
Copyright 2018, Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
