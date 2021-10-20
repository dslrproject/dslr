import torch
import torch.nn as nn
# from my_eval import *
import torch.utils.data
import torch
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from utils import * 
from models import *
import argparse
import tqdm
import random

parser = argparse.ArgumentParser(description='VAE training of LiDAR')

parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--z_dim',              type=int,   default=128,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achliargsas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--pose_dim',           type=int,   default=128,             help='size of the pose vector')
parser.add_argument('--output_layers',      type=int,   default=100,            help='number of layers')
parser.add_argument('--optimizer',                      default='adam',         help='optimizer to train with')
parser.add_argument('--lr',                 type=float, default=0.002,          help='learning rate')
parser.add_argument('--beta1',              type=float, default=0.5,            help='momentum term for adam')
parser.add_argument('--epochs',             type=int,   default=10,             help='number of epochs to train for')
parser.add_argument('--weights',            type=str,   default='',             requried=True help='location of the discrimintor weights to test with')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()



#-----------------------------------------------------------------------------------------------------------

# class scene_discriminator(nn.Module):
#     def __init__(self, pose_dim, nf=1024):
#         super(scene_discriminator, self).__init__()
#         self.pose_dim = pose_dim
#         self.main = nn.Sequential(
#                 nn.Linear(pose_dim*2, nf),
#                 nn.BatchNorm1d(nf),
#                 nn.Sigmoid(),
#                 nn.Linear(nf, nf),
#                 nn.BatchNorm1d(nf),
#                 nn.Sigmoid(),
#                 nn.Linear(nf, nf),
#                 nn.BatchNorm1d(nf),
#                 nn.Sigmoid(),
#                 nn.Linear(nf, 1),
#                 nn.Sigmoid(),
#                 )



class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=256):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
                # nn.Dropout(p=0.5),
                nn.Linear(pose_dim*2, int(pose_dim)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim), int(pose_dim/2)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim/2), int(pose_dim/4)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/4),int(pose_dim/8)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/8),1),
                nn.Sigmoid()
               
                )



# class scene_discriminator(nn.Module):
#     def __init__(self, pose_dim, nf=256):
#         super(scene_discriminator, self).__init__()
#         self.pose_dim = pose_dim
#         self.main = nn.Sequential(
#                 # nn.Dropout(p=0.5),
#                 nn.Linear(pose_dim*2, int(pose_dim)),
#                 nn.Sigmoid(),
#                 # nn.Dropout(p=0.5),
#                 nn.Linear(int(pose_dim), int(pose_dim/2)),
#                 nn.Sigmoid(),
#                 # nn.Dropout(p=0.5),
#                 nn.Linear(int(pose_dim/2), int(pose_dim/4)),
#                 nn.Sigmoid(),
#                 nn.Linear(int(pose_dim/4),int(pose_dim/8)),
#                 nn.Sigmoid(),
#                 nn.Linear(int(pose_dim/8),int(pose_dim/16)),
#                 nn.Sigmoid(),
#                 nn.Linear(int(pose_dim/16),1),
#                 nn.Sigmoid()
#                 )



    def forward(self, input1,input2):
        # print('forward')
        # print(input[0].shape)                    #torch.Size([1,256])
        # print((torch.cat(input, 0)).shape)       #torch.Size([2,256])
        # print((torch.cat(input, 0).view(-1, self.pose_dim*2)).shape)   #torch.Size([1,512])
        # exit()
        # print('Cat Shape')
        # print(torch.cat((input1, input2),1).shape)
        output = self.main(torch.cat((input1, input2),1).view(-1, self.pose_dim*2))
        # print('Output in forward ',output)
        return output

#-----------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#class to load own dataset

class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self,continousStatic,continousStatic1,pairStatic,pairDynamic):
        super(Pairdata, self).__init__()
        self.continousStatic  = continousStatic
        self.continousStatic1 = continousStatic1
        self.pairStatic       = pairStatic
        self.pairDynamic      = pairDynamic

    def __len__(self):
        return self.pairStatic.shape[0]

    def __getitem__(self, index):
        
        return index, self.continousStatic[index], self.continousStatic1[index], self.pairStatic[index],self.pairDynamic[index]

# dataloaders1 = DataLoader(DummyDataset(0, 9), batch_size=2, shuffle=True)
# dataloaders2 = DataLoader(DummyDataset(0, 9), batch_size=2, shuffle=True)



# ------------------------------------------------------------------------------







#---------to port gretHidden rpe code here just for testing, but can make itermannent---------------------
process_input = from_polar if args.no_polar else lambda x : x


#img is [1,60,512,4]

#the decoder takes in the input in xyz format      [something:3:40:256]
#the decoder gives the recon in polar coorrdinates [something:2:40:256]

def getHiddenRep(img):
    # img=torch.Tensor(img)
    # img=img.cuda()
    # print(type(img))
    # print(img.shape)
    # exit(1)
    # img   = preprocess(img).astype('float32')       #[1,2,40,256]
    
    img=torch.Tensor(img)
    img=img.cuda()
    # print('In getHiddenRep')
    # print(img.shape) #[1,2,40,256]
    # print(((process_input(img).detach())).shape)
    # exit(1)
    # show_pc_lite(process_input(img))
    # exit(1)
    # print(img.shape)                              [64,2,40,256]
    # print(((process_input(img))).shape)  #            [64,3,40,256]
    # print(((process_input(img))[0:1,:,:,:]).shape)  [1,3,40,256]
    # print(show_pc_lite((process_input(img))[0:1,:,:,:]))
    recon, kl_cost, hidden_z = model(process_input(img))
    # print(recon.detach().shape)                   [64,2,40,256]
    # print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
    # print((recon[0:1,:,:,:]).shape)               ([1, 2, 40, 256]
    # show_pc_lite(from_polar((recon.detach())))
    return hidden_z











#-----------------------------------------------------------------------------------------------------------


#make the input x like it has 2 frames, one static and the other corresponding dynamic
#one taining is with (x1 with x1)--output 0     (x1 with x2)--output 1

# x has shape [2,2,60,512,4]






args = parser.parse_args()
print('---Creating Model---')
model = VAE(args).cuda()
# network=torch.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation_exp/plsDoNotDel/runs/test/models/gen_1769.pth')
# network=torch.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation_exp/plsDoNotDel/runs/test/models/weightsOnFull100-184-st+dynamicBrokeInBetween/gen_1953.pth')
# model.load_state_dict(network)

network=torch.load(args.weights)
model.load_state_dict(network['state_dict_gen'])
model.eval()
print('---VAE Model Created and Loaded---')















print('---Creating and Loading Discriminator weights---')
netC = scene_discriminator(args.pose_dim, 256).cuda()
#In order to train the scene_discrminator
# networkDisc=torch.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation_exp/plsDoNotDel/disc_10.pth')
netC.load_state_dict(network['state_dict_disc'])
netC.eval()
print('---Discriminator created and Loaded---')




# ------------------------------------------------------------

bce_criterion = nn.BCELoss()
netC.cuda()

#...............................................................



print('---Loading the Testing Datset--')

#here I could have dome num_batches instead if num_batches-1 but don't know why in the last batch :- looks Like it was like the modulus last part and has zeros that were 
#casuing issues


continous_val         = np.load('/home/prashant/Desktop/static_continous/16.npy')
continous_val1        = np.load('/home/prashant/Desktop/static_continous/20.npy')
pair_static_val       = np.load('/home/prashant/Desktop/purestatic/19.npy')
pair_dynamic_val      = np.load('/home/prashant/Desktop/puredynamic/17.npy')

continous_val         = preprocess(continous_val).astype('float32')
continous_val1        = preprocess(continous_val1).astype('float32')
pair_static_val       = preprocess(pair_static_val).astype('float32')
pair_dynamic_val      = preprocess(pair_dynamic_val).astype('float32')

val_data             = Pairdata(continous_val,continous_val1,pair_static_val,pair_dynamic_val)
val_loader           = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,drop_last=True)


one=[]
two=[]

   

def plot(out):
    l=list(out)
    x=random.randint(0,100)
    plt.figure()
    plt.hist(l, bins=10)
    plt.savefig(str(x)+'.jpg')
    plt.show()




for i, valPair in enumerate(val_loader):
    model.eval()
    loss=0

    stContinous1 = valPair[1]
    # print(stContinous1.shape)
    # raise SystemError
    stContinous2 = valPair[2]
    stPartPair   = valPair[3]
    dyPartPair   = valPair[4]
    # show_pc_lite((from_polar(stPartPair.cuda()))[0:1,:,:,:])
    # show_pc_lite((from_polar(dyPartPair.cuda()))[0:1,:,:,:])
    hiddenRep1=getHiddenRep(stContinous1).detach()
    hiddenRep2=getHiddenRep(stContinous2).detach()
    hiddenRep3=getHiddenRep(stPartPair).detach()
    hiddenRep4=getHiddenRep(dyPartPair).detach()

    # print(hiddenRep1.shape)
    # print(hiddenRep2.shape)
    # print(hiddenRep3.shape)
    # print(hiddenRep4.shape)

    out1 = netC(hiddenRep1,hiddenRep2)
    for i in out1.detach().cpu():
        one.append(i)
        # print(out1)

    out2 = netC(hiddenRep3,hiddenRep4)
    for i in out2.detach().cpu():
        two.append(i)
        # print(out2)

        

plot(one)     
plot(two)



