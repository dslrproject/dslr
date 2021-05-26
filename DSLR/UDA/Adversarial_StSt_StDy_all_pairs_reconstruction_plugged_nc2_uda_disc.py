#!/usr/bin/env python

import math
import torch
import torch.nn as nn

# from my_eval import *
import torch.utils.data
import torch
import sys
import matplotlib.pyplot as plt
from utils import *
from models512 import *
from torch.utils.data import DataLoader, Dataset
import argparse
import tensorboardX
import tqdm
import random


parser = argparse.ArgumentParser(description="VAE training of LiDAR")

parser.add_argument(
    "--batch_size", type=int, default=128, help="size of minibatch used during training"
)
parser.add_argument(
    "--use_selu", type=int, default=0, help="replaces batch_norm + act with SELU"
)
parser.add_argument(
    "--base_dir", type=str, default="runs/test", help="root of experiment directory"
)
parser.add_argument(
    "--no_polar",
    type=int,
    default=1,
    help="if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)",
)
parser.add_argument(
    "--z_dim",
    type=int,
    default=160,
    help="size of the bottleneck dimension in the VAE, or the latent noise size in GAN",
)
parser.add_argument(
    "--autoencoder",
    type=int,
    default=1,
    help="if True, we do not enforce the KL regularization cost in the VAE",
)
parser.add_argument(
    "--atlas_baseline",
    type=int,
    default=0,
    help="If true, Atlas model used. Also determines the number of primitives used in the model",
)
parser.add_argument(
    "--panos_baseline",
    type=int,
    default=0,
    help="If True, Model by Panos Achliargsas used",
)
parser.add_argument(
    "--kl_warmup_epochs",
    type=int,
    default=150,
    help="number of epochs before fully enforcing the KL loss",
)
parser.add_argument("--pose_dim", type=int, default=160, help="size of the pose vector")
parser.add_argument("--output_layers", type=int, default=100, help="number of layers")
parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--beta1", type=float, default=0.1, help="momentum term for adam")
parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs to train for"
)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

writer = tensorboardX.SummaryWriter(
    log_dir=os.path.join(args.base_dir, "2-Adversarial-segmented-static-mmd")
)
writes = 0
ns = 16
alpha = 10
# With the weights of  gen_100.pth are the saved odd pth weight
# wih the gen_698 are the ones saved with even weights

FILEPATH = "/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pair_transform_dynseg/preprocessed/"
maybe_create_dir(args.base_dir + "2-Adversarial-segmented-static-mmd")
# -----------------------------------------------------------------------------------------------
# All helper functions


def print_and_log_scalar(writer, name, value, write_no, end_token=""):
    if isinstance(value, list):
        if len(value) == 0:
            return
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name)
    name += " " * zeros
    print("{} @ write {} = {:.4f}{}".format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)


def getHiddenRep(img):
    img = torch.Tensor(img)
    img = img.cuda()
    recon, kl_cost, hidden_z = model(process_input(img))
    return hidden_z


def freezeNetwork(model):
    for param in model.parameters():
        param.requires_grad = False


def freezeDecoder(model):
    print(model.decode)
    for param in model.decode.parameters():
        param.requires_grad = False
    print("Decoder Freezed")


# ------------------------------------------------------------------------------
# class to load own dataset


class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, pairStatic, pairDynamic, pairStaticForRecon, pairKittyDynamic):
        super(Pairdata, self).__init__()

        self.pairStatic = pairStatic
        self.pairDynamic = pairDynamic
        self.pairStaticForRecon = pairStaticForRecon
        self.pairKittyDynamic = pairKittyDynamic

    def __len__(self):
        return self.pairDynamic.shape[0]

    def __getitem__(self, index):

        return (
            index,
            self.pairStatic[index],
            self.pairDynamic[index],
            self.pairStaticForRecon[index],
            self.pairKittyDynamic[index],
        )


class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=256):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(pose_dim * 2, int(pose_dim)),
            nn.Sigmoid(),
            # nn.Dropout(p=0.5),
            nn.Linear(int(pose_dim), int(pose_dim / 2)),
            nn.Sigmoid(),
            # nn.Dropout(p=0.5),
            nn.Linear(int(pose_dim / 2), int(pose_dim / 4)),
            nn.Sigmoid(),
            nn.Linear(int(pose_dim / 4), int(pose_dim / 8)),
            nn.Sigmoid(),
            nn.Linear(int(pose_dim / 8), 1),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2):
        output = self.main(torch.cat((input1, input2), 1).view(-1, self.pose_dim * 2))
        return output


# ------------------------------------------------------------------------------

process_input = from_polar if args.no_polar else lambda x: x
loss_fn = lambda a, b: (a - b).abs().sum(-1).sum(-1).sum(-1)
# -------------------------------------------------------------------------------

# load the files in dataloaders beforehand


def load(npyList):
    retList = []
    index = 0
    for file1 in npyList:
        index += 0
        for file2 in npyList[index:]:
            print("Pair ", file1, file2)

            dataset_pair_static = np.load(FILEPATH + "s/" + file1 + ".npy")[
                :, :, :40, ::2
            ].astype("float32")
            # data_pair_static          = preprocess(dataset_pair_static).astype('float32')
            # del dataset_pair_static
            # print('Deleted 1')

            dataset_pair_dynamic = np.load(FILEPATH + "d/" + file2 + ".npy")[
                :, :, :40, ::2
            ].astype("float32")
            # data_pair_dynamic         = preprocess(dataset_pair_dynamic).astype('float32')
            # del dataset_pair_dynamic
            # print('Deleted 2')

            dataset_pair_st_for_recon = np.load(FILEPATH + "s/" + file2 + ".npy")[
                :, :, :40, ::2
            ].astype("float32")

            # data_pair_st_for_recon = preprocess(dataset_pair_st_for_recon).astype('float32')
            # del dataset_pair_st_for_recon
            # print('Deleted 3')

            dataset_pair_kitty = np.load(FILEPATH + "dk/" + file2 + ".npy")[
                :, :, :40, ::2
            ].astype("float32")

            print("After Loading")

            train_data = Pairdata(
                dataset_pair_static,
                dataset_pair_dynamic,
                dataset_pair_st_for_recon,
                dataset_pair_kitty,
            )

            del dataset_pair_static, dataset_pair_dynamic, dataset_pair_st_for_recon
            train_loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=False, drop_last=True
            )

            retList.append(train_loader)
            del train_loader
    print(retList)
    return retList


# npyList=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
npyList = ["0", "1", "2", "3"]

# npyList = ['0']
npyList = load(npyList)


# ------------------------------------------------------------------------------


# this is the encoder: oe for the static files and one for the dynamic files

args = parser.parse_args()
print("---Creating Model---")
modelStatic = VAE(args).cuda()
# network698 = torch.load('gen_698.pth')
# network1= torch.load('ATM-TRACK/normalAE-[2-40-128]-hiddenDim-128/gen_693.pth')

# network = torch.load(
#    "../dual_loss_stst_stdy_random/runs/test/3-disc-with-segmentedStatic-data/gen_26.pth"
# )

network = torch.load("runs/test/1-mmd-disc-with-normalStatic-data/gen_10.pth")

modelStatic.load_state_dict(network["state_dict_gen"])
freezeNetwork(modelStatic)


modelDynamic = VAE(args).cuda()
modelDynamic.load_state_dict(network["state_dict_gen"])
freezeDecoder(modelDynamic)

optimizerDynamic = optim.Adam(
    modelDynamic.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-5
)
# optimizerDynamic.load_state_dict(network_reload['optimizer'])

# this is the dis criminoator for the stst stdy frame types
print("---Creating Discriminator---")
netC = scene_discriminator(args.pose_dim, 256)
netC.cuda()
netC.load_state_dict(network["state_dict_disc"])
freezeNetwork(netC)
print("---Models Created and Loaded---")


print("---Discriminator created ---")

bce_criterion = nn.BCELoss()
bce_criterion.cuda()


print("---Optimizer and Loss Created---")

# ---------------- optimizers ----------------------------------


pair_static_val = np.load(FILEPATH + "s/4.npy")[:, :, :40, ::2].astype("float32")
pair_dynamic_val = np.load(FILEPATH + "d/4.npy")[:, :, :40, ::2].astype("float32")
pair_dynamic_kitty = np.load(FILEPATH + "dk/4.npy")[:, :, :40, ::2].astype("float32")
pair_static_recon_val = pair_static_val

print(pair_static_val.shape)
print(pair_dynamic_val.shape)
print(pair_dynamic_kitty.shape)

print("Preprocessing Validation")
# pair_static_val         = preprocess(pair_static_val).astype('float32')
# pair_dynamic_val        = preprocess(pair_dynamic_val).astype('float32')
# pair_static_recon_val   = preprocess(pair_static_recon_val).astype('float32')


val_data = Pairdata(
    pair_static_val, pair_dynamic_val, pair_static_recon_val, pair_dynamic_kitty
)
val_loader = DataLoader(
    val_data, batch_size=args.batch_size, shuffle=False, drop_last=True
)

del pair_static_val, pair_dynamic_val, pair_static_recon_val, val_data
process_input = from_polar if args.no_polar else lambda x: x


def calc_mmd_loss(x, y, alpha=0.001):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    B = x.shape[0]

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K = torch.exp(-alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(-alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(-alpha * (rx.t() + ry - 2 * zz))

    beta = 1.0 / (B * (B - 1))
    gamma = 2.0 / (B * B)

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)


for epoch in tqdm.trange(args.epochs):
    print("Epoch ", epoch)
    indexOfNpyList = -1
    lmbda = 2 / (1 + math.exp(-10 * (epoch) / 1000)) - 1
    for train_loader in npyList:

        # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
        process_input = from_polar if args.no_polar else lambda x: x
        loss_ae_, loss_disc_ = [[] for _ in range(2)]

        for i, trainPair in enumerate(train_loader):
            modelDynamic.train()
            loss_disc = 0
            loss_disc_noise = 0
            modelDynamic.zero_grad()

            stPartPair = trainPair[1].cuda()
            dyPartPair = trainPair[2].cuda()
            dyPartPairForRecom = trainPair[3].cuda()
            dyPartPairKitty = trainPair[4].cuda()

            recon0, kl_cost0, z_hid0 = modelStatic(process_input(stPartPair))
            recon1, kl_cost1, z_hid1 = modelDynamic(process_input(dyPartPair))
            recon2, kl_cost2, z_hid2 = modelDynamic(process_input(dyPartPairKitty))

            hiddenRep1 = z_hid0
            hiddenRep2 = z_hid1
            hiddenRep3 = z_hid2

            # ---------------------------------------------------------------------------------------
            # Training the adversarial with the clean dynamic lidar frame
            out1 = netC(hiddenRep1, hiddenRep2)
            target1 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(1)
            loss_st_dy = bce_criterion(out1, Variable(target1))
            loss_disc += loss_st_dy

            # Reconstruction Loss
            loss_recon = loss_fn(recon1[:, :, :40, :], (dyPartPairForRecom))
            loss_recon = (loss_recon).mean(dim=0)
            loss_disc /= args.batch_size

            # ---------------------------------------------------------------------------------------
            # MMD loss between Dynamic Carla and Dynamic Kitty. Discriminator
            loss_mmd = calc_mmd_loss(
                z_hid1.view(args.batch_size, -1), z_hid2.view(args.batch_size, -1)
            )

            # ---------------------------------------------------------------------------------------
            # Training Adv with the Discriminator loss between static carla and dynamic kitti.
            out1 = netC(hiddenRep1, hiddenRep3)
            target1 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(1)
            loss_st_dy = bce_criterion(out1, Variable(target1))
            loss_disc += loss_st_dy

            print(loss_mmd.item())
            loss_final = (alpha * loss_disc) + loss_recon + (lmbda * loss_mmd)
            loss_final.backward(retain_graph=True)

            loss_ae_ += [loss_recon.item()]
            loss_disc_ += [loss_disc.item()]

            optimizerDynamic.step()

            writes += 1
        mn = lambda x: np.mean(x)
        print_and_log_scalar(
            writer, "DiscriminatorTrain-discLoss", mn(loss_disc_), writes
        )
        print_and_log_scalar(
            writer, "DiscriminatorTrain-reconLoss", mn(loss_ae_), writes
        )
        # print_and_log_scalar(writer, 'Total trainig Loss', (loss_final + loss_final_noise)/2, writes)

    loss_ae_, loss_disc_, loss_disc_ad_, loss_mmd_ = [[] for _ in range(4)]

    for i, valPair in enumerate(val_loader):
        modelStatic.eval()
        modelDynamic.eval()
        netC.eval()
        loss_disc = 0
        stPartPair = valPair[1].cuda()
        dyPartPair = valPair[2].cuda()
        stPartforDyRecon = valPair[3].cuda()
        dyPartPairKitty = valPair[4].cuda()

        recon0, kl_cost0, z_hid0 = modelStatic(process_input(stPartPair))
        recon1, kl_cost1, z_hid1 = modelDynamic(process_input(dyPartPair))
        recon2, kl_cost2, z_hid2 = modelDynamic(process_input(dyPartPairKitty))

        hiddenRep1 = z_hid0
        hiddenRep2 = z_hid1
        hiddenRep3 = z_hid2

        out1 = netC(hiddenRep1, hiddenRep2)
        target1 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(1)
        target2 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0)
        loss_st_dy_ad = bce_criterion(out1, Variable(target1))
        loss_st_dy = bce_criterion(out1, Variable(target2))

        # ------------------------------------------------------
        # Reconstruction Loss checking for static reconstrcution
        loss_recon = loss_fn(recon1[:, :, :40, :], (stPartforDyRecon))

        loss_recon = loss_recon.mean(dim=0)

        loss_st_dy = loss_st_dy.mean(dim=0)

        loss_st_dy_ad = loss_st_dy_ad.mean(dim=0)

        loss_mmd = calc_mmd_loss(
            z_hid1.view(args.batch_size, -1), z_hid2.view(args.batch_size, -1)
        )

        # ------------------------------------------------------

        loss_ae_ += [loss_recon.item()]

        loss_disc_ += [loss_st_dy.item()]

        loss_disc_ad_ += [loss_st_dy_ad.item()]

        loss_mmd_ += [loss_mmd.item()]

    writes += 1
    mn = lambda x: np.mean(x)
    print_and_log_scalar(
        writer, "DiscriminatorTestAdver", mn(loss_disc_ad_), writes
    )  # should go down
    print_and_log_scalar(
        writer, "DiscriminatorTest", mn(loss_disc_), writes
    )  # should go up
    print_and_log_scalar(writer, "ReconstructionTestForEncoder", mn(loss_ae_), writes)
    print_and_log_scalar(writer, "MMDLOss", mn(loss_mmd_), writes)

    # print('Loss for Training in batch ',batch, 'is', epoch_sd_loss/batchSize)
    # save the model

    if epoch % 3 == 0:
        # torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch+1978)))
        state = {
            "epoch": epoch + 1,
            "state_dict_gen_st": modelStatic.state_dict(),
            "state_dict_gen_dy": modelDynamic.state_dict(),
            "optimizer": optimizerDynamic.state_dict(),
        }
        torch.save(
            state,
            os.path.join(
                args.base_dir,
                "2-Adversarial-segmented-static-mmd/gen_{}.pth".format(epoch),
            ),
        )

