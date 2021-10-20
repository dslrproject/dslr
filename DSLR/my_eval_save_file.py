#This is the new eval that saves to disk the files are recosntructing them
#instead of using my_Eval.py now use this for evaluating a recosntruction

from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm,trange 
from utils512 import * 

from models512 import *

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=1024,           help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=1,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--data',               type=str,   default='',             required=True, help='Location of the data to train')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

FILEPATH = args.data


def save_on_disk(static,dynamic,reconstructedStaticWhole):
	np.savez('SaveRecons/static',static.cpu())
	np.savez('SaveRecons/dynamic',dynamic)
	np.savez('SaveRecons/recon_st',reconstructedStaticWhole)





# Encoder must be trained with all types of frames,dynmaic, static all
model = VAE(args).cuda()

#network = torch.load("runs/gen_45.pth")
network = torch.load(args.ae_weight)
#network = torch.load("runs/test/4-Adversarial-segmented-static-mmd/gen_81.pth")
#network=torch.load('runs/test4-Adversarial-normal-static/gen_6.pth')
# network=torch.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation_exp/plsDoNotDel/runs/test/models/weightsOnFull100-184-st+dynamicBrokeInBetween/gen_1913.pth')

# network1=torch.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation_exp/plsDoNotDel/runs/test/models/gen-128-hidden-dimension_till700_vgoodres/gen_698.pth')

model.load_state_dict(network['state_dict_gen_dy'])
model.eval()







#print the correspoding data for refrence
dataset_corr_st   = np.load(FILEPATH + 's/tests4.npy')[:,:,:40,::2].astype('float32')
# dataset_corr_st   = preprocess(dataset_corr_st).astype('float32')


dataset_dynamic   = np.load(FILEPATH + 'd/testd4.npy')[:,:,:40,::2].astype('float32')
# dataset_val   = preprocess(dataset_corr_st).astype('float32')


#dataset_val   = np.load('/home/prashant/P/ATM/Code/lidar_generation/lidar_generation/kitti_data/lidar_test.npz')



# print(dataset_val.shape)    #(154,60,512,4)
# exit(1)
# print(dataset_val[0:1,:,:,:].shape)  #[1,60,512,4] 
# dataset_val   = preprocess(dataset_val).astype('float32')
# print("Here")
# print(type(dataset_val))
# exit(1)
# dataset_val1   = preprocess(dataset_val[0:1,:,:,:]).astype('float32')
# print(dataset_val1.shape)      #(1,2,40,256)

# print(dataset_val.shape)          #(152,2,40,256)
# exit(1)

val_loader    = torch.utils.data.DataLoader(dataset_dynamic, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
process_input = from_polar if args.no_polar else lambda x : x
# print 





recons=[]
original=[]


for i, img in enumerate(val_loader):
	# print(i)
	# if(i!=10):
	# 	continue

	# print(type(img))								
	# exit(1)
	img = img.cuda()
	# print(type(img))								#torch.Tensor
	# print(img.shape)       						#[65,2,40,256]
 		
	print(((process_input(img))[0:1,:,:,:]).shape)  #[1,3,40,256] 
	print(show_pc_lite((from_polar(img))[0:1,:,:,:]))

	recon, kl_cost,z = model(process_input(img))
	print(type(recon))
	recons=recon
	original=img
	# print(recon.detach().shape)  					[64,2,40,256]
	# print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
	# print((recon[0:1,:,:,:]).shape)				([1, 2, 40, 256]
	print(show_pc_lite((from_polar(recon[0:1,:,:,:]).detach())))
	break
	



# dataset_corr_st=torch.Tensor(dataset_corr_st)


recons_temp=np.array(recons.detach().cpu())    			#(64, 2, 40, 256)
original_temp=np.array(original.detach().cpu())    	    #(64, 2, 40, 256)
dataset_corr_st=torch.Tensor(dataset_corr_st).cuda()



# print('Saving')
# print(recons_temp.shape)
# print(original_temp.shape)
# print(dataset_corr_st.shape)
# #save_on_disk(dataset_corr_st,original_temp,recons_temp)
if not os.path.exists('samplesVid-seg/reconstructed/'): 
    os.makedirs('samplesVid-seg/reconstructed/')
    os.makedirs('samplesVid-seg/original/')
    os.makedirs('samplesVid-seg/originalst/')


for frame_num in tqdm(range(recons_temp.shape[0])):
	frame=from_polar(recons[frame_num:frame_num+1,:,:,:]).detach().cpu()
	plt.figure()
	plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
	plt.savefig('samplesVid-seg/reconstructed/'+str(frame_num)+'.png') 
	# plt.show()

for frame_num in tqdm(range(original_temp.shape[0])):
	frame=from_polar(original[frame_num:frame_num+1,:,:,:]).detach().cpu()
	plt.figure()
	plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
	plt.savefig('samplesVid-seg/original/'+str(frame_num)+'.png') 
	# plt.show()



i=0
for frame_num in tqdm(range(dataset_corr_st.shape[0])):
	if(i<1024):
		i+=1
		frame=from_polar(dataset_corr_st[frame_num:frame_num+1,:,:,:]).detach().cpu()
		plt.figure()
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid-seg/originalst/'+str(frame_num)+'.png') 
	else:
		break



