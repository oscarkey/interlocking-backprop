# Interlocking Backpropagation: Improving depthwise model-parallelism

Code associated with the paper: [Interlocking Backpropagation: Improving depthwise model-parallelism](https://arxiv.org/abs/2010.04116) (Aidan N. Gomez, Oscar Key, Stephen Gou, Nick Frosst, Jeff Dean, Yarin Gal)

This is an implementation of various model-parallel training schemes in PyTorch.
Its primary aim is to be a research platform for studying these approaches, but it might also be a useful base for a production implementation.

This is the code for the ResNet experiments in the paper.
The Transformer experiments are implemented in a proprietary codebase, so unfortunately we are unable to release them.

## Citation
If this code has been useful in your research, please add a citation:
```
@article{gomez2020interlocking,
      title={Interlocking Backpropagation: Improving depthwise model-parallelism},
      author={Aidan N. Gomez and Oscar Key and Stephen Gou and Nick Frosst and Jeff Dean and Yarin Gal},
      year={2020},
      eprint={2010.04116},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Usage
First, read the paper to understand the types of models and optimization schemes that this library implements.

`examples/train_resnet.py` demonstrates the features of the library, by showing how to train a ResNet using various model-parallel schemes.

As this library is designed for experimenting with different training schemes, it supports both simulated and true model-parallism.
In simulated mode, only a single GPU is required thus allowing the user to run a large number of different experiments.
True model-parallism can be enabled by calling `InterlockingBackpropModel.enable_model_parallel()`. This will distribute the model across all available GPUs, placing one component on each GPU (hence the number of components must match the number of GPUs available).

## Current limitations
 - The library uses multi-threading rather than multi-processing, which could result in slower performance due to GIL contention.
 In our testing with 4 GPUs, threads spent only a few percent of their time waiting on the GIL, but it might be a problem for larger numbers of GPUs.


## Development
To set up the environment: `conda env update -f environment.yaml`

To run the tests: `pytest tests`

To format the code: `black **/*.py`

To run the type checker: `mypy interlocking_backprop` (there are still a couple of errors)
