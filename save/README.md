This is the default location of results from training a code.
Using the configuration files provided in the [conf](../conf) directory,
results will be saved according to the following directory hierarchy:
```
<dataset>/<base model>/k<parameter k>/<loss function>/<encoder architecture>
```

The following files will be saved:
* `{train, val}_loss.txt`: Average loss for each epoch.
* `{train, val}_reconstruction_accuracy.txt`: Reconstruction accuracy attained
  in each epoch of training. Recall from the paper that reconstruciton-accuracy
  is defined as the accuracy of decoded reconstructions with respect to the
  classifications of the underlying base model.
* `{train, val}_overall_accuracy.txt`: Overall-accuracy attained in each epoch
  of training. Recall from the paper that overall-accuracy is defined as the
  accuracy of decoded reconstructions with respect to the true labels associated
  with a particular dataset.
* `current.pth`: PyTorch checkpoint containing the latest snapshots of the
  encoder and decoder parameters, encoder and decoder optimizers, and current epoch.
* `best.pth`: The state dict among all epoch so far that has achieved the highest
  reconstruction-accuracy on the validation dataset.
