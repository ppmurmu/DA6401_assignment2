# Part-A

## Pre-requisite
- Install packages:
  ```bash
  pip install -r requirements.txt
- Download the ``train-a.py`` script.
- Download the dataset [Click here]([URL](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)) and keep it in the same directory as ``train-a.py`` file.
- Run the script
  ```bash
  python train-a.py
## 🛠️ Command-Line Arguments

| Argument              | Type     | Default               | Description                                                   |
|-----------------------|----------|------------------------|---------------------------------------------------------------|
| `--dir`               | `str`    | `./inaturalist_12K/`   | Path to dataset directory                                     |
| `--wandb_project`     | `str`    | `DA6401-A2`            | Weights & Biases project name                                 |
| `--wandb_entity`      | `str`    | `cs24m033-iit-madras`  | Weights & Biases entity name                                  |
| `--epoch`             | `int`    | `1`                    | Number of training epochs                                     |
| `--num_filters`       | `str`    | `[128,128,64,64,32]`   | List of filters per conv layer (stringified list)             |
| `--filter_sizes`      | `str`    | `[3,5,5,7,7]`          | List of filter sizes (stringified list)                       |
| `--weight_decay`      | `float`  | `0.0`                  | Weight decay (L2 regularization)                              |
| `--dropout`           | `float`  | `0.0`                  | Dropout rate                                                  |
| `--learning_rate`     | `float`  | `1e-4`                 | Learning rate for the optimizer                               |
| `--activation`        | `str`    | `gelu`                 | Activation function (`relu`, `gelu`, `silu`, `mish`)          |
| `--optimiser`         | `str`    | `adam`                 | Optimizer (`adam`, `nadam`, `rmsprop`)                        |
| `--batch_norm / --no_batch_norm` | `bool` | `True`        | Enable or disable batch normalization                         |
| `--augment / --no_augment`       | `bool` | `True`        | Enable or disable data augmentation                           |
| `--batch_size`        | `int`    | `64`                   | Batch size (options: 32, 64)                                  |
| `--dense_layer`       | `int`    | `512`                  | Number of neurons in the dense (fully connected) layer        |


