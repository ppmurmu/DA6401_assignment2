# Part-B

## 🔧 Pre-requisite
- Install packages:
  ```bash
  pip install -r requirements.txt
- Download the ``train-b.py`` script.
- Download the dataset [Click here]([URL](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)) and keep it in the same directory as ``train-b.py`` file.
- Run the script
  ```bash
  python train-b.py
## 🛠️ Command-Line Arguments

| Argument              | Type     | Default               | Description                                                   |
|-----------------------|----------|------------------------|---------------------------------------------------------------|
| `--dir`               | `str`    | `./inaturalist_12K/`   | Path to dataset directory                                     |
| `--wandb_project`     | `str`    | `DA6401-A2`            | Weights & Biases project name                                 |
| `--wandb_entity`      | `str`    | `cs24m033-iit-madras`  | Weights & Biases entity name                                  |
| `--epoch`             | `int`    | `1`                    | Number of training epochs                                     |
| `--strategy`          | `str`    | `freeze_all`           | Strategy for fine-tuning (`freeze_all`, `unfreeze_last`, etc.)|
| `--learning_rate`     | `float`  | `1e-4`                 | Learning rate for the optimizer                               |
