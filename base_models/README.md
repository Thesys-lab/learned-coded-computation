# Base Models
This directory contains implementations of the [Base-MLP](base_mlp.py),
[ResNet-18](resnet.py) and [Logistic Regression](logistic.py) base models
used in evaluation in the paper. Model files containing the trained parameters
for each base model used in the paper's evaluation are under the
[base_model_trained_files](../base_model_trained_files) directory.

### Adding a new base model
If you'd like to add a new base model for evaluation, you need to make to
perform the following steps:
1. Implement the base model in PyTorch. Your base model must inherit from
   `torch.nn.Module` and must implement the `__init__` and `forward` methods.
   The `forward` method must take just one parameter, a batch of samples over
   which a forward pass is performed.
2. Train your base model and save the base model's state dictionary to a file.
   This may be done using `torch.save(my_base_model.state_dict(), "my_file.t7")`
3. Create a new configuration file that specifically changes the following parameters:
   1. `BaseModel`: Specify the classpath of your base model in the "class" field
   and any arguments required for the `__init__` function in the "args" field.
   2. `base_model_file`: Specify the path to the state dictionary saved in step (2).
   3. `base_model_input_size`: Specify the input dimensions expected of inputs to
   the `forward` method of your base model.

   There are more details on other configuration parameters in the
   [conf](../conf) directory.
