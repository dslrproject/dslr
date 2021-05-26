<p align="center"><img src='./images/cover.png' width="60%"/></p>

# DSLR: Dynamic to Static LiDAR Scan Reconstruction Using Adversarially Trained Autoencoder

### Prerequisites
- Linux or macOS
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3.6
- PyTorch 1.7.1 & torchvision 0.8.2 (https://pytorch.org/)
- [TensorboardX 2.0](https://github.com/lanpa/tensorboard-pytorch)

### Install
- Clone this repo:
```
git clone https://github.com/sabyasachis/temp-dslr-repo.git
cd DSLR/
```
- Install PyTorch and other dependencies mentioned in the Prerequisites
  - For pip users, please type the command `pip install -r requirements.txt`
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`

## Dataset
The steps to download the ARD-16 (Ati Realworld Dataset) paired correspondence dataset have been mentioned in this [Readme](https://github.com/sabyasachis/temp-dslr-repo/blob/main/Data_Generation/README.md)



## Training Example

## Testing Example

## Training Options

- Common for `autoencoder.py`, `discriminator.py` and `adversarial_training.py`
  - `--base_dir` for setting the root of experiment directory.
  - `--batch_size` for setting the size of minibatch used during training.
  - `--use_selu` replaces batch_norm + act with SELU
  - `--no_polar` If True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)
  - `--lr` for setting the learning rate value
  - `--z_dim` for setting the size of the bottleneck dimension in the VAE, or the latent noise size in GAN
  - `--autoencoder` If True, we do not enforce the KL regularization cost in the VAE
  - `--atlas_baseline` If True, then the AtlasNet model is used. It also determines the number of primitives used in the model.
  - `--panos_baseline` If True, then the Model by Panos Achlioptas is used.
  - `--kl_warmup_epochs` for setting the number of epochs before fully enforcing the KL loss.

- Paired discriminator
  - `--pose_dim` for setting the size of the pose vector.
  - `--output_layers` for setting the number of layers.
  - `--optimizer` for setting the optimizer to train with.
  - `--beta1` for setting the momentum term for Adam
  - `--epochs` for setting the number of epochs to train the discriminator.

- Adversarial Training of Autoencoder
  - `--pose_dim` for setting the size of the pose vector.
  - `--output_layers` for setting the number of layers.
  - `--optimizer` for setting the optimizer to train with.
  - `--beta1` for setting the momentum term for Adam
  - `--epochs` for setting the number of epochs to train the discriminator.
