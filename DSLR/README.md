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

For our training we used a subset of CARLA-64 dataset available in the DATASET section due to GPU limitations. The same set were used to report all the results. 
It is encouraged to use all the dataset from (0-15) available and re-run our method on that dataset to get numbers using the whole dataset, if possible to get the numbers using all the datasets. 
We will report the number on all the dataset in due course of time.

The steps to download the ARD-16 (Ati Realworld Dataset) paired correspondence dataset have been mentioned in this [Readme](https://github.com/sabyasachis/temp-dslr-repo/blob/main/Data_Generation/README.md)



## Training Example
To start with trainig the model we need to train 3 modules. 
1st Module: Autoencoder
```
python autoencoer.py --data [all lidar scan in npy format made available in dataset seciton]
```
Use the "tensorbard --logdir runs" to visulaize the training.
Choose an epoch at which the training loss plateaus and validation loss starts to increase. Take the weights from the 'runs' folder

2nd Module : Discriminator
```
python discriminator.py --data [path to the corresponsing pair. Use the above location. The code handles the pairing]
```
The training saturates very fast for the discriminator. Keep a test [static dynamic pair] and test the model. Check if it classfies [st-st] pairs as ~1 and [st-dy] pairs as ~0.
Use a specific epochs weights according to the check above.

Test the discriminator weights using a separate test [static-dynamic pair] as below:
```
python my_discrimintor_test_rand_stst_stdy.py  [location of the discriminator weights]
```

3rd Module: Adversarial Training
Use the weights from the discriminator for the adversarial model:

```
python  adversarial_traning.py [path of corresponding pais. use the previous path]
```
Training saturates in the first 50 epochs. Check the tensorboard logs for details. Pick an epoch where the val and trainig error are both low.

The model output from here can be used for dynamic to static conversions: Test this model using the evaluation/testing code codel.

```
python my_eval_save_file --data [location of the data] --ae_weights [weight of the final adversarial module]
```
Use the output weight from the adversarila module as the weights for the dynamic to static conversion weights: Pick a static-dynamic pair and use the below commands to retrieve the reconstructed output static using the adversarial model




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
