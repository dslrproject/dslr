#!/usr/bin/env python
# 7th August 2020, This is the correct file for the discriminator training without any noise 

import torch
import torch.nn as nn
# from my_eval import *
import torch.utils.data
import torch
import sys
import matplotlib.pyplot as plt
from utils512 import * 
from models512 import *
from torch.utils.data import DataLoader, Dataset
import argparse
import tensorboardX
import tqdm
import random
import gc


parser = argparse.ArgumentParser(description='VAE training of LiDAR')

parser.add_argument('--batch_size',         type=int,   default=32,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test/',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=1,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achliargsas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--pose_dim',           type=int,   default=160,            help='size of the pose vector')
parser.add_argument('--output_layers',      type=int,   default=100,            help='number of layers')
parser.add_argument('--optimizer',                      default='adam',         help='optimizer to train with')
parser.add_argument('--lr',                 type=float, default=0.0006,         help='learning rate')
parser.add_argument('--beta1',              type=float, default=0.1,            help='momentum term for adam')
parser.add_argument('--epochs',             type=int,   default=50,             help='number of epochs to train for')
parser.add_argument('--data',               type=str,   default='',             required = 'True , 'help='Location of the dataset')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

DISCRIMINATOR_RUN = "discriminator_git_v1"

writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, DISCRIMINATOR_RUN))
writes = 0
ns     = 16
alpha  = 10
# FILE_PATH = '/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pSysFinalSlicData/test/test'

FILE_PATH = args.data
maybe_create_dir(args.base_dir+DISCRIMINATOR_RUN)

#-----------------------------------------------------------------------------------------------
#logging


def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)








# ------------------------------------------------------------------------------
#process input

process_input = from_polar if args.no_polar else lambda x : x

def getHiddenRep(img):
    img=torch.Tensor(img)
    img=img.cuda()
    recon, kl_cost, hidden_z = model(process_input(img))
    return hidden_z



#-----------------------------------------------------------------------------------------------------------
#declare classes

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
        return self.pairDynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.continousStatic[index], self.continousStatic1[index], self.pairStatic[index],self.pairDynamic[index]


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


    def forward(self, input1,input2):
        output = self.main(torch.cat((input1, input2),1).view(-1, self.pose_dim*2))
        return output


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: #or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#-------------------------------------------------------------------------------------------------------


# Stuff for model making and BPlearning etc

if args.atlas_baseline or args.panos_baseline:
    loss_fn = get_chamfer_dist()
else:
    loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)


args = parser.parse_args()
print('---Creating Model---')
model = VAE(args).cuda()
network=torch.load("/home/saby/old_sabyasachi/Desktop/Pr/plsDoNotDelVAE/runs/test/autoencoder_git_v1/gen_10.pth")
model.load_state_dict(network['state_dict'])
# model.eval()



print('---Creating Discriminator---')
netC = scene_discriminator(args.pose_dim, 256)
netC.cuda()
# netC.load_state_dict(network['state_dict_disc'])
print('---Models Created and Loaded---')
#In order to train the scene_discrminator


# netC.train()
print('---Discriminator created ---')

bce_criterion = nn.BCELoss()
bce_criterion.cuda()


print('---Optimizer and Loss Created---')

# ---------------- optimizers ---------------------------------------------------------------



if args.optimizer == 'adam':
    args.optimizer = optim.Adam
elif args.optimizer == 'rmsprop':
    args.optimizer = optim.RMSprop
elif args.optimizer == 'sgd':
    args.optimizer = optim.SGD
else:
  raise ValueError('Unknown optimizer: %s' % opt.optimizer)

optimizerDual = optim.Adam(list(netC.parameters())+list(model.parameters()), lr=args.lr, betas=(args.beta1, 0.999),weight_decay=1e-5)
# optimizerDual.load_state_dict(network['optimizer'])



#....................load dataset...............................

def load(npyList):
    retList=[]
    indexOfNpyList = 0
    for file1 in npyList:
    	indexOfNpyList += 0
    	for file2 in npyList[indexOfNpyList:]:
            print('Pair ', file1,file2)
            dataset_static_continous = np.load(FILE_PATH+'s'+file1+'.npy')[:,:,0:40,::2].astype('float32')
            #dataset_static_continous  = preprocess(dataset_static_continous).astype('float32')

            dataset_static_continous1 = np.load(FILE_PATH+'s'+file2+'.npy')[:,:,0:40,::2].astype('float32')
            #dataset_static_continous1 = preprocess(dataset_static_continous1).astype('float32')


            data_pair_static      = dataset_static_continous

            dataset_pair_dynamic     = np.load(FILE_PATH+'d'+file2+'.npy')[:,:,0:40,::2].astype('float32')
            
            
            train_data= Pairdata(dataset_static_continous , dataset_static_continous1 , data_pair_static , dataset_pair_dynamic)
            del dataset_static_continous,dataset_static_continous1,data_pair_static, dataset_pair_dynamic
            
            train_loader=DataLoader(train_data, batch_size=args.batch_size, shuffle=False,drop_last=True)
            retList.append(train_loader)
            
            
            gc.collect()
    print(retList)
    
    return retList



# npyList=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
#npyList = ['0','1','2','3']
npyList = ['0', "1"]

npyList = load(npyList)



#npyList=['0','1','2','3','4']

continous_val        = np.load(FILE_PATH+'s4.npy')[:,:,0:40,::2].astype('float32')
continous_val1       = continous_val
pair_static_val      = continous_val
pair_dynamic_val     = np.load(FILE_PATH+'d4.npy')[:,:,0:40,::2].astype('float32')



#continous_val        = preprocess(continous_val).astype('float32')
#continous_val1       = preprocess(continous_val1).astype('float32')
#pair_static_val      = continous_val
#pair_dynamic_val     = preprocess(pair_dynamic_val).astype('float32')
val_data             = Pairdata(continous_val,continous_val1,pair_static_val,pair_dynamic_val)
del continous_val, continous_val1,pair_static_val, pair_dynamic_val
val_loader           = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,drop_last=True)
del val_data



indexOfNpyList=-1


# All these variables are used by testing and validation phase and are set to 0 before the startting of each phase
loss_ae = 0
loss_disc = 0
loss_ae_noise = 0
loss_disc_noise = 0
final_loss = 0
final_loss_noise = 0



listLen = len(npyList)

for epoch in tqdm.trange(args.epochs):
    print('Epoch ', epoch)
    indexOfNpyList=-1



    for train_loader  in npyList:
	    	
        # indexOfNpyList+=1
        # for file2 in npyList[indexOfNpyList:]:
            # print('\n\n---Loading the Datset--->',file1,file2)
            # print('Here')
            # dataset_static_continous = np.load('/home/prashant/Desktop/v3Data/s/'+file1+'.npy')
            # dataset_static_continous1 = np.load('/home/prashant/Desktop/v3Data/s/'+file2+'.npy')
            # dataset_pair_static      = np.load('/home/prashant/Desktop/v3Data/s/'+file1+'.npy')
            # dataset_pair_dynamic     = np.load('/home/prashant/Desktop/v3Data/d/'+file2+'.npy')
            # # dataset_val   = np.load('/home/prashant/P/carla-0.8.4/PythonClient/_out_npy/carlaData/0_2/val.npy')
            # print('After Loading')
            # # print('Continous Shape',dataset_static_continous.shape)
            # # print('Pair Shape',dataset_pair_dynamic.shape)
            # dataset_static_continous  = preprocess(dataset_static_continous).astype('float32')
            # dataset_static_continous1 = preprocess(dataset_static_continous1).astype('float32')
            # data_pair_static          = preprocess(dataset_pair_static).astype('float32')
            # data_pair_dynamic         = preprocess(dataset_pair_dynamic).astype('float32')

            # train_data= Pairdata(dataset_static_continous,dataset_static_continous1,data_pair_static,data_pair_dynamic)
            # train_loader=DataLoader(train_data, batch_size=args.batch_size, shuffle=False,drop_last=True)

            
            # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
            process_input = from_polar if args.no_polar else lambda x : x

            for i, trainPair in enumerate(train_loader):
                model.train()
                netC.train()
                loss_ae = 0
                loss_disc = 0
                loss_ae_noise = 0
                loss_disc_noise = 0
                final_loss = 0
                final_loss_noise = 0
                netC.zero_grad()
                model.zero_grad()


                stContinous1 = trainPair[1].cuda()
                # print(stContinous1.shape)
                # raise SystemError
                stContinous2 = trainPair[2].cuda()
                stPartPair   = trainPair[3].cuda()
                dyPartPair   = trainPair[4].cuda()
                dyPartPair_noise = add_noise(dyPartPair, random.random() * 0.4)
            	# show_pc_lite((from_polar(noisy_img[0:1,:,:,:]).detach()))
                dyPartPair_noise = (randomly_remove_entries(dyPartPair_noise, random.random() * 0.4 ))
            	
            	
            	#training the autoencoder------------------------------------
                recon0, kl_cost0, z_hid0 = model(process_input(stContinous1))
                recon1, kl_cost1, z_hid1 = model(process_input(stContinous2))
                recon2, kl_cost2, z_hid2 = model(process_input(stPartPair))
                recon3, kl_cost3, z_hid3 = model(process_input(dyPartPair))
#                 recon3_noise, kl_cost3_noise, z_hid3_noise = model(process_input(dyPartPair_noise))

                loss_recon0 = loss_fn(recon0[:,:,:40,:512], (stContinous1))
                loss_recon1 = loss_fn(recon1[:,:,:40,:512], (stContinous2))
                loss_recon2 = loss_fn(recon2[:,:,:40,:512], (stPartPair))
                loss_recon3 = loss_fn(recon3[:,:,:40,:512], (dyPartPair))
#                 loss_recon3_noise = loss_fn(recon3_noise[:,:,:40,:512], (dyPartPair))
                gc.collect()


                print('epoch %s' % epoch)
                loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]


                # if args.autoencoder:
                #     kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                # else:
                #     kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                #                     torch.clamp(kl_cost, min=5)

                # loss0 = (loss_recon).mean(dim=0)
                # loss1 = (loss_recon).mean(dim=0)
                # loss2 = (loss_recon).mean(dim=0)
                # loss3 = (loss_recon).mean(dim=0)
                # print('here')
                loss_ae          += (loss_recon0+loss_recon1+loss_recon2 + loss_recon3).mean(dim=0)
#                 loss_ae_noise    += (loss_recon0+loss_recon1+loss_recon2 + loss_recon3_noise).mean(dim=0)
                # print(shape(loss_recon0+loss_recon1))
                # print('here1')
                loss_ae    /= 4
#                 loss_ae_noise /= 4
                # print('here1')
                loss_   =   [loss_ae.item()]
                # elbo_    += [elbo.item()]
                # kl_cost_ += [kl_cost.mean(dim=0).item()]
                # kl_obj_  += [kl_obj.mean(dim=0).item()]
                # recon_   += [loss_recon.mean(dim=0).item()]


                #------------------------------------------------------------




                
                # print('here2')
                # show_pc_lite((from_polar(stPartPair.cuda()))[0:1,:,:,:])
                # show_pc_lite((from_polar(dyPartPair.cuda()))[0:1,:,:,:])
                hiddenRep1=z_hid0
                hiddenRep2=z_hid1
                hiddenRep3=z_hid2
                hiddenRep4=z_hid3
#                 hiddenRep4_noise=z_hid3_noise
                # print('here3')
                
              

                #--------------------------------------------------------------
                #The training without noise in the dynamic
                out1 = netC(hiddenRep1,hiddenRep2)

                target1 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(1)

                
                loss_static = bce_criterion(out1, Variable(target1))
                loss_disc+=loss_static

                # print('here4')
                
                out2 = netC(hiddenRep3,hiddenRep4)
                target2 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0)
                loss_dynamic = bce_criterion(out2, Variable(target2))
                loss_disc+=loss_dynamic
            

                # print('here3')
                # raise SystemError
                # loss_recon = loss_fn(recon, img[:,:,0:24,:])
                loss_disc /= (args.batch_size*2)

                final_loss = loss_ae + alpha * loss_disc

                writes += 1
                # mn = lambda x : np.mean(x)
                final_loss.backward(retain_graph = True)



#                 #--------------------------------------------------------------
#                  #The training with noise in the dynamic
#                 out2_noise    = netC(hiddenRep3,hiddenRep4_noise)
#                 target2_noise = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0)
#                 loss_dynamic_noise = bce_criterion(out2_noise, Variable(target2_noise))
#                 loss_disc_noise +=(loss_static + loss_dynamic_noise)
            

#                 # print('here3')
#                 # raise SystemError
#                 # loss_recon = loss_fn(recon, img[:,:,0:24,:])
#                 loss_disc_noise /= (args.batch_size*2)

#                 final_loss_noise = loss_ae_noise + alpha*loss_disc_noise

#                 writes += 1
#                 # mn = lambda x : np.mean(x)
#                 final_loss_noise.backward()
#                 #---------------------------------------------------------------


                optimizerDual.step()

                print_and_log_scalar(writer, 'train AE/loss', (loss_ae )/(listLen), writes)
                print_and_log_scalar(writer, 'train Discriminator/loss', (loss_disc )/(listLen), writes)
                del stContinous1, stContinous2, stPartPair, dyPartPair, recon0, recon1, recon2, recon3, hiddenRep1, hiddenRep2, hiddenRep3, hiddenRep4
                del out1, out2, target1, target2, loss_static, loss_dynamic
                #del loss_ae, loss_ae_noise, loss_disc, loss_disc_noise, final_loss, final_loss_noise, kl_cost0, kl_cost1, kl_cost2, kl_cost3, z_hid0, z_hid1, z_hid2, z_hid3
                
                gc.collect()
                torch.cuda.empty_cache() 
                


    


    
    for i, valPair in enumerate(val_loader):
                loss_ae = 0
                loss_disc = 0
                loss_ae_noise = 0
                loss_disc_noise = 0
                final_loss = 0 
                final_loss_noise = 0
                model.eval()
                netC.eval()
                stContinous1 = valPair[1].cuda()
                # print(stContinous1.shape)
                # raise SystemError
                stContinous2 = valPair[2].cuda()
                stPartPair   = valPair[3].cuda()
                dyPartPair   = valPair[4].cuda()
#                 dyPartPair_noise = add_noise(dyPartPair, random.random() * 0.4)
            	# show_pc_lite((from_polar(noisy_img[0:1,:,:,:]).detach()))
#                 dyPartPair_noise = (randomly_remove_entries(dyPartPair_noise, random.random() * 0.4 ))
            	
            	
            	#training the autoencoder------------------------------------
                recon0, kl_cost0, z_hid0 = model(process_input(stContinous1))
                recon1, kl_cost1, z_hid1 = model(process_input(stContinous2))
                recon2, kl_cost2, z_hid2 = model(process_input(stPartPair))
                recon3, kl_cost3, z_hid3 = model(process_input(dyPartPair))
#                 recon3_noise, kl_cost3_noise, z_hid3_noise = model(process_input(dyPartPair_noise))

                loss_recon0 = loss_fn(recon0[:,:,:40,:512], (stContinous1))
                loss_recon1 = loss_fn(recon1[:,:,:40,:512], (stContinous2))
                loss_recon2 = loss_fn(recon2[:,:,:40,:512], (stPartPair))
                loss_recon3 = loss_fn(recon3[:,:,:40,:512], (dyPartPair))
#                 loss_recon3_noise = loss_fn(recon3_noise[:,:,:40,:512], (dyPartPair))
                gc.collect()


                
                loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]


                # if args.autoencoder:
                #     kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                # else:
                #     kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                #                     torch.clamp(kl_cost, min=5)

                # loss0 = (loss_recon).mean(dim=0)
                # loss1 = (loss_recon).mean(dim=0)
                # loss2 = (loss_recon).mean(dim=0)
                # loss3 = (loss_recon).mean(dim=0)
                # print('here')
                loss_ae          += (loss_recon0 + loss_recon1 + loss_recon2 + loss_recon3).mean(dim=0)
#                 loss_ae_noise    += (loss_recon0 + loss_recon1 + loss_recon2 + loss_recon3_noise).mean(dim=0)
                # print(shape(loss_recon0+loss_recon1))
                # print('here1')
                loss_ae    /= 4
#                 loss_ae_noise /= 4
                # print('here1')
                loss_   =   [loss_ae.item()]
                # elbo_    += [elbo.item()]
                # kl_cost_ += [kl_cost.mean(dim=0).item()]
                # kl_obj_  += [kl_obj.mean(dim=0).item()]
                # recon_   += [loss_recon.mean(dim=0).item()]


                #------------------------------------------------------------




                
                # print('here2')
                # show_pc_lite((from_polar(stPartPair.cuda()))[0:1,:,:,:])
                # show_pc_lite((from_polar(dyPartPair.cuda()))[0:1,:,:,:])
                hiddenRep1=z_hid0
                hiddenRep2=z_hid1
                hiddenRep3=z_hid2
                hiddenRep4=z_hid3
#                 hiddenRep4_noise=z_hid3_noise
                # print('here3')
                
              

                #--------------------------------------------------------------
                #The training without noise in the dynamic
                out1 = netC(hiddenRep1,hiddenRep2)

                target1 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(1)

                
                loss_static = bce_criterion(out1, Variable(target1))
                loss_disc+=loss_static

                # print('here4')
                
                out2 = netC(hiddenRep3,hiddenRep4)
                target2 = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0)
                loss_dynamic = bce_criterion(out2, Variable(target2))
                loss_disc+=loss_dynamic
            

                # print('here3')
                # raise SystemError
                # loss_recon = loss_fn(recon, img[:,:,0:24,:])
                loss_disc /= (args.batch_size*2)

                

                writes += 1
                # mn = lambda x : np.mean(x)
                



#                 #--------------------------------------------------------------
#                  #The training with noise in the dynamic
#                 out2_noise    = netC(hiddenRep3,hiddenRep4_noise)
#                 target2_noise = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0)
#                 loss_dynamic_noise = bce_criterion(out2_noise, Variable(target2_noise))
#                 loss_disc_noise += (loss_static + loss_dynamic_noise)
            

#                 # print('here3')
#                 # raise SystemError
#                 # loss_recon = loss_fn(recon, img[:,:,0:24,:])
#                 loss_disc_noise /= (args.batch_size*2)

#                 #final_loss_noise = loss_ae_noise + alpha*loss_disc_noise

#                 writes += 1
#                 # mn = lambda x : np.mean(x)
#                 #---------------------------------------------------------------

        # mn = lambda x : np.mean(x)


    print_and_log_scalar(writer, 'AE validation Reconstruction', (loss_ae)/(listLen), writes)
    print_and_log_scalar(writer, 'Disc validation', (loss_disc )/(listLen), writes)   
    gc.collect()




        # print('Loss for Training in batch ',batch, 'is', epoch_sd_loss/batchSize)
        
        # save the model
    



    if(epoch%1==0):
        # torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch+1978)))
        state = {'epoch': epoch + 1, 'state_dict_gen': model.state_dict(),'state_dict_disc': netC.state_dict(),'optimizer': optimizerDual.state_dict()}
        torch.save(state, os.path.join(args.base_dir, '{}/disc_{}.pth'.format(DISCRIMINATOR_RUN, epoch)))
