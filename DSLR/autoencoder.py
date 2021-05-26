import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
from torchsummary import summary
from tqdm import trange
from utils512 import * 
from models512 import * 
# from geomloss import SamplesLoss
import random
import numpy as np
import os


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=512,            help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test/',   help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=1,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')

# --------------------------------------------------------------------------------------------
args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)
DATA = '/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pSysFinalSlicData/test/test'
RUN_SAVE_PATH = "autoencoder_git_v1"
maybe_create_dir(args.base_dir+RUN_SAVE_PATH)



# the baselines are very memory heavy --> we split minibatches into mini-minibatches
if args.atlas_baseline or args.panos_baseline: 
    """ ed on 12 Gb GPU for z_dim in [128, 256, 512] """ 
    bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
    factor = args.batch_size // bs
    args.batch_size = bs
    is_baseline = True
    args.no_polar = 1
    print('using batch size of %d, ran %d times' % (bs, factor))
else:
    factor, is_baseline = 1, False


#----------------------------------------------------------------------------------------------


# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# construct model and ship to GPU
model = VAE(args).cuda()

#the npy file have the size as (64,512,4)
#however in the preprocess() we changes it to 60,256
#print(summary(model, input_size=(2, 40, 512)))
#exit(1)



#----------------------------------------------------------------------------------------------
#Load Data

def load(npyList):
    retList=[]
    for file in npyList:
        print(file)
        data_train = np.load(DATA+file)
        data_train = data_train[:,:,5:45,::2].astype('float32')
        #data_train   = preprocess(data_train).astype('float32')
        #print(data_train.shape)#
        #dataset_train = data_train[:,:,:,::2]
        #print(dataset_train.shape)
        #del data_train
        # raise SystemError
        # print(dataset_train.shape)
        # print('After preprocessing')
        # print(dataset_train.shape)
        train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
        #del data_train
        retList.append(train_loader)
    print(retList)
    return retList




# npyList=['preprocessed/d/0.npy','preprocessed/s/seg0.npy','preprocessed/d/1.npy','preprocessed/s/seg1.npy','preprocessed/d/2.npy','preprocessed/s/seg2.npy','preprocessed/d/3.npy','preprocessed/s/seg3.npy','preprocessed/d/4.npy']

#npyList=data=['s/0.npy','s/1.npy','s/2.npy','s/s35.npy','d/0.npy','d/1.npy','d/2.npy','d/d35.npy']

npyList=['s0.npy']
# npyList=['d10.npy']

# npyList=['s15.npy','230.npy']

#npyList=['s/0.npy','s/1.npy','s/2.npy','s/3.npy','s/4.npy','s/5.npy','s/6.npy','s/7.npy','s/8.npy','s/9.npy','s/10.npy','s/11.npy','s/12.npy','s/13.npy','d/0.npy','d/1.npy','d/2.npy','d/3.npy','d/4.npy','d/5.npy','d/6.npy','d/7.npy','d/8.npy','d/9.npy','d/10.npy','d/11.npy','d/12.npy','d/13.npy']

#npyList=['s/0.npy']
npyList1 =load(npyList)
print(npyList1)

#----------------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------------


# print(model)
model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr) 

# network=torch.load('runs//models/gen_399.pth')

# model.load_state_dict(network['state_dict'])
# optim.load_state_dict(network['optimizer'])
print('Previous weight and optimizer loaded')




# Logging
# maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, RUN_SAVE_PATH))
writes = 0
ns     = 16



data_val   = np.load(DATA + 'd14.npy')[:,:,5:45,::2].astype('float32')
#dataset_val   = preprocess(data_val).astype('float32')

# print(dataset_val.shape)
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                         shuffle=True, num_workers=4, drop_last=True)
val_loader    = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size,shuffle=False, num_workers=4, drop_last=False)
print("val data loader")
#del data_val
# build loss function
if args.atlas_baseline or args.panos_baseline:
    loss_fn = get_chamfer_dist()
else:
    loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
    # loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=.99, backend="online")

   
if __name__ == "__main__":
    # VAE training
    # ------------------------------------------------------------------------------
    rangee=150 if args.autoencoder else 300
    for epoch in (trange(700)):
        # dataset preprocessing
        for loader in npyList1:
            print('New Data Module loaded')
            print('loading data')
            # dataset_train = file
            # print('After Loading')
            # print(dataset_train.shape)

            # if args.debug: 
            #     dataset_train, dataset_val = dataset_train[:128], dataset_val[:128]

            
            print('epoch %s' % epoch)
            train_loader = loader
            model.train()
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
                                        
            # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
            process_input = from_polar if args.no_polar else lambda x : x

            for i, img in enumerate(train_loader):
                img = img.cuda()
                model.zero_grad()
                #Process original input
                recon, kl_cost, _ = model(process_input(img))
                loss_recon = loss_fn(recon[:,:,0:40,:], img)
                # loss_recon.backward(retain_graph=True)



                #-------------------------------------------------------------------------------------
                #Code to train with noisy removed entries
                #Process noisy original input: ie some entries removed

                # model.zero_grad()
                # noisy_img = process_input(randomly_remove_entries(img,random.random()*0.2))
                # # show_pc_lite((from_polar(noisy_img[0:1,:,:,:]).detach()))
                # noisy_img = add_noise(noisy_img, random.random()*0.2)

                # recon_noise, kl_cost, _ = model(noisy_img)
                # loss_recon_noise = loss_fn(recon_noise[:,:,0:40,:], img)
                # loss_recon_noise = loss_recon_noise.mean(dim = 0)
                # loss_recon_noise.backward()
                
                #-------------------------------------------------------------------------------------
                
                # print('Inp')
                # print(process_input(img[:,:,0:24,:]).shape)
                # print('Recon')
                # print(recon.shape)
                # raise SystemError
                # loss_recon = loss_fn(recon, img[:,:,0:24,:])
                	

                if args.autoencoder:
                    kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                else:
                    kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                                torch.clamp(kl_cost, min=5)

                loss = (kl_obj  + loss_recon).mean(dim=0)
                elbo = (kl_cost + loss_recon).mean(dim=0)

                loss_    += [loss.item()]
                elbo_    += [elbo.item()]
                kl_cost_ += [kl_cost.mean(dim=0).item()]
                kl_obj_  += [kl_obj.mean(dim=0).item()]
                recon_   += [loss_recon.mean(dim=0).item()]

                # baseline loss is very memory heavy 

                loss.backward()
                optim.step()
               
        
            writes += 1
            mn = lambda x : np.mean(x)
            print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
            print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
            print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
            print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
            print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
            
        # save some training reconstructions
        # if epoch % 10 == 0:
        #      recon = recon[:ns].cpu().data.numpy()
        #      with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
        #          np.save(f, recon)

        #      # print('saved training reconstructions')
             
        
        # ing loop
        # --------------------------------------------------------------------------

        loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
        with torch.no_grad():
            model.eval()
            if epoch % 1 == 0:
                # print(' set evaluation')
                for i, img in enumerate(val_loader):
                    img = img.cuda()

                    #Process original input
                    recon, kl_cost,_ = model(process_input(img))
                    loss_recon = loss_fn(recon[:,:,0:40,:], img)



                    #-----------------------------------------------------------------
                    #Process noisy original input: ie some entries removed

                    # noisy_img = process_input(randomly_remove_entries(img,0))
                   	# noisy_img = add_noise(noisy_img, 0)
                   	# recon_noise, kl_cost,_ = model(noisy_img)
                    # loss_recon_noise = loss_fn(recon_noise[:,:,0:40,:], img)

                    #-----------------------------------------------------------------


                    if args.autoencoder:
                        kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                    else:
                        kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                                        torch.clamp(kl_cost, min=5)
                    
                    loss = (kl_obj  + loss_recon).mean(dim=0)
                    elbo = (kl_cost + loss_recon).mean(dim=0)

                    loss_    += [loss.item()]
                    elbo_    += [elbo.item()]
                    kl_cost_ += [kl_cost.mean(dim=0).item()]
                    kl_obj_  += [kl_obj.mean(dim=0).item()]
                    recon_   += [loss_recon.mean(dim=0).item()]

                print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)
                print_and_log_scalar(writer, 'valid/elbo', mn(elbo_), writes)
                print_and_log_scalar(writer, 'valid/kl_cost_', mn(kl_cost_), writes)
                print_and_log_scalar(writer, 'valid/kl_obj_', mn(kl_obj_), writes)
                print_and_log_scalar(writer, 'valid/recon', mn(recon_), writes)
                loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

                    # if epoch % 10 == 0:
                    #     with open(os.path.join(args.base_dir, 'samples/_{}.npz'.format(epoch)), 'wb') as f: 
                    #         recon = recon[:ns].cpu().data.numpy()
                    #         np.save(f, recon)
                    #         # print('saved  recons')
                       
                    #     sample = model.sample()
                    #     with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
                    #         sample = sample.cpu().data.numpy()
                    #         np.save(f, recon)
                        
                    #     # print('saved model samples')
                        
                    # if epoch == 0: 
                    #     with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
                    #         img = img.cpu().data.numpy()
                    #         np.save(f, img)
                        
                        # print('saved real LiDAR')

        if(epoch%5==0):
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': optim.state_dict()}
            torch.save(state, os.path.join(args.base_dir + RUN_SAVE_PATH, 'gen_{}.pth'.format(epoch)))