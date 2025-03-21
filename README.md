# Downstream Task Guided Masking Learning in Masked Autoencoders Using Multi-Level Optimization

**Han Guo, Ramtin Hosseini, Ruiyi Zhang, Sai Ashish Somayajula, Ranak Roy Chowdhury, Rajesh K. Gupta, Pengtao Xie**

Published on *Transactions on Machine Learning Research (TMLR)*.

[![arXiv](https://img.shields.io/badge/arXiv-2402.18128-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2402.18128)


## Introduction

Masked Autoencoder (MAE) is a notable method for self-supervised pretraining in visual representation learning. It operates by randomly masking image patches and reconstructing these masked patches using the unmasked ones. A key limitation of MAE lies in its disregard for the varying informativeness of different patches, as it uniformly selects patches to mask. To overcome this, some approaches propose masking based on patch informativeness. However, these methods often do not consider the specific requirements of downstream tasks, potentially leading to suboptimal representations for these tasks. In response, we introduce the Multi-level Optimized Mask Autoencoder (MLO-MAE), a novel framework that leverages end-to-end feedback from downstream tasks to learn an optimal masking strategy during pretraining. Our experimental findings highlight MLO-MAE's significant advancements in visual representation learning. Compared to existing methods, it demonstrates remarkable improvements across diverse datasets and tasks, showcasing its adaptability and efficiency.

![main.pdf](main.pdf)


## MLO-MAE Pre-training and Fine-tuning

### Overview
This repository contains Python scripts for pretraining and fine-tuning models on CIFAR-10/100 and ImageNet-1K datasets using our proposed MLO-MAE method. It supports Data Distributed Parallel (DDP) for efficient training across multiple GPUs and integrates with WandB for experiment tracking and logging.

### Prerequisites
Before running the scripts, ensure you have an appropriate Python environment setup. You can create an environment using Conda or pip with the provided `environment.yml` or `requirements.txt` files, respectively.

#### Using Conda:
```bash
conda env create -f environment.yml
conda activate mae
```

#### Using pip:
```bash
pip install -r requirements.txt
```

### MLO-MAE pretraining
To start retraining your models using the MLO-MAE method, use the following scripts for CIFAR-10, CIFAR-100, and ImageNet-1K datasets:

- For CIFAR-10:
```bash
torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:23428 \
--rdzv_id 326 \
main_cifar10.py
```

- For CIFAR-100:
```bash
torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:23428 \
--rdzv_id 326 \
main_cifar100.py
```

- For ImageNet-1K:
```bash
torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=localhost:23428 \
--rdzv_id 326 \
main_imagenet.py \
--epochs 50 \
--blr 1.5e-4 \
--batch_size 128 \
--output_dir ./output \
--log_dir ./output \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--weight_decay 0.05 \
--unroll_steps_pretrain 2 \
--unroll_steps_finetune 1 \
--unroll_steps_mask 1 \
--base_finetune_lr 5e-4 \
--finetune_batchsize 64 \
--base_masking_lr 5e-5 \
--masking_batchsize 64 \
--world_size 2 \
--data_path ./data
```

Refer to each script's documentation for a complete list of supported arguments. Note to reproduce the results reported in the paper please use the same experimental settings and hypereparameters as reported in the main paper and appendix. In our experiments, we use 2 A100 GPUs on pretraining MLO-MAE.

### Fine-tuning
After retraining, you can fine-tune the models using the following scripts:

- For CIFAR-10:
```bash
python finetune_cifar10.py --arg1 <value1> --arg2 <value2> ... 
```

- For CIFAR-100:
```bash
python finetune_cifar100.py --arg1 <value1> --arg2 <value2> ... 
```

- For ImageNet-1K:
```bash
python finetune_imagenet.py --arg1 <value1> --arg2 <value2> ... 
```

### Data Distributed Parallel (DDP)
The scripts are compatible with DDP for distributed training. 

### Logging with WandB
To log your experiments with WandB, ensure you have an account and obtain your API key. Use the `--wandb_key` argument to provide your API key when running the scripts.


### Script Arguments Summary

#### Training Scripts
The training scripts (`main_cifar10.py`, `main_cifar100.py`, `main_imagenet.py`) support a variety of arguments for customization, including:

- `--batch_size`: Specify the batch size for training.
- `--epochs`: Set the number of training epochs.
- `--model`: Choose the model architecture.
- `--input_size`: Define the input size for the model.
- `--lr`: Learning rate for the optimizer.
- `--data_path`: Path to the dataset.
- `--output_dir`: Directory to save model checkpoints.
- `--log_dir`: Directory for logging outputs.
- `--wandb_key`: WandB API key for logging.

Additional arguments like `--mask_ratio`, `--weight_decay`, and `--warmup_epochs` allow further customization of the training process.

#### Fine-tuning Scripts
The fine-tuning scripts (`finetune_cifar10.py`, `finetune_cifar100.py`, `finetune_imagenet.py`) include arguments for fine-tuning the pre-trained models, such as:

- `--finetune`: Path to the pre-trained model for fine-tuning.
- Similar to the training scripts, they support arguments for batch size, learning rate, model selection, and data handling.

Both sets of scripts also include advanced options for distributed training, data augmentation techniques, and specific adjustments for the training/fine-tuning process.

For detailed information on all supported arguments, please refer to the individual script files.


## Citation

```latex
@article{
}
```



## Contact Us

If you have any questions, please contact:

- Han Guo: h5guo@ucsd.edu
