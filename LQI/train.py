"""
Refactoring required
"""


import argparse
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
from tqdm import tqdm, trange
import os, shutil
from torchsummary import summary
import pickle
import gc
from utils import * 
from models import * 
from torch.optim.lr_scheduler import StepLR
import random

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,                           help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=128,             help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dataset', type = str, default = 'carla')
# ------------------------ARGS stuff----------------------------------------------
args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# the baselines are very memory heavy --> we split minibatches into mini-minibatches
if args.atlas_baseline or args.panos_baseline: 
    """ Tested on 12 Gb GPU for z_dim in [128, 256, 512] """ 
    bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
    factor = args.batch_size // bs
    args.batch_size = bs
    is_baseline = True
    args.no_polar = 1
    print('using batch size of %d, ran %d times' % (bs, factor))
else:
    factor, is_baseline = 1, False
gc.collect()

# --------------------- Setting same seeds for reproducibility ------------------------------
# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# -------------------- MODEL BUILDING -------------------------------------------------------
# construct model and ship to GPU

model = CNNIQAnet(args).cuda()
#network=torch.load("models_noise_predict/gen_162.pth")
#model.load_state_dict(network)
#model = Unet(args).cuda()
print(model)
print(summary(model, input_size=(2, 40, 512)))
gc.collect()
# assert False

model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1)#, amsgrad=True) 
scheduler = StepLR(optim, step_size=3, gamma=0.5)
# MODEL_PATH = '/home/saby/Projects/ati/ati_motors/adversarial_based/occlusion_segpaint/second_full_data/models/gen_20.pth'
# if(os.path.exists(MODEL_PATH)):
#     network_state_dict = torch.load(MODEL_PATH)
#     model.load_state_dict(network_state_dict)
#     print("Previous weights loaded from {}".format(MODEL_PATH))
# else:
# 	print("Starting from scratch")
#assert False
gc.collect()

# -------------------- TENSORBOARD SETUP FOR LOGGING -------------------------------------------
# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models_noise_predict_smol_test'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB_predict_test'))
writes = 0
ns     = 16

# -------------------- DATASET SETUP ----------------------------------------------------------
def getint(name):
    try:
        return int(name.split('.')[0])
    except Exception as e:
        print("Error occured while trying to read {}".format(name))
    return None


class add_noise_partial():
    def __init__(self):
        pass
        #self.noise = [x/100 for x in range(0,50,5)]

    def __call__(self, sample):
        idx, img, mask = sample
        #img = from_polar_np(img.numpy())
        #img = torch.from_numpy(img)
        masksh = mask.shape
        imgsh = img.shape
        noise = np.random.randint(0, 100)/100
        noise_tensor = torch.zeros_like(img).normal_(0, noise)
        means = img.view((2, -1)).mean(-1)
        stds  = img.view((2, -1)).std(-1)
        #print(means[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).shape)
        means, stds = [x.unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]
        # normalize data
        norm = (img - means) / (stds + 1e-9)
        norm = norm + noise_tensor
        # unnormalize
        unnorm = norm * (stds + 1e-9) + means
        noise_tensor = noise_tensor*(stds+1e-9) + means
        unnorm = unnorm.reshape(imgsh)
        #mask = mask.reshape(masksh)i
        noise_tensor = noise_tensor.reshape(imgsh)
        norm = norm.reshape(imgsh)
        #norm = norm.reshape(imgsh)
        return idx, norm, np.ceil(10*noise)

class Pairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset1,dataset2,transform=None):
        super(Pairdata, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transform
    
    def __len__(self):
        return self.dataset1.shape[0]

    def __getitem__(self, index):
        img = torch.from_numpy(self.dataset1[index][:,5:45,::2])
        mask = torch.from_numpy(self.dataset2[index][:,5:45,::2])
        if self.transform:
            index, img, mask = self.transform((index,  img, mask))    
        return index, img, mask



PREPROCESS_FRESHLY = False

# After preprocessing path

# input to model 
if args.dataset == 'kitti':
    DYNAMIC_PREPROCESS_DATA_PATH = "testing_data_preprocessed/kitti_preprocessed_run"
    PATH_FOR_MASK_SAVING_AND_LOADING = "testing_data_preprocessed/kitti_mask"
elif args.dataset == 'carla':
    DYNAMIC_PREPROCESS_DATA_PATH = "carla_data/dynamic_preprocessed_new"
    PATH_FOR_MASK_SAVING_AND_LOADING = "carla_data/dynamic_mask_npy_new"
if PREPROCESS_FRESHLY:
    # Un preprocessed path
    # input to model
    DYNAMIC_TRAIN_FOLDER_PATH = "/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pair_transform_dynseg_all/static_out_npy/"
    # output of the model
    
    DYNAMIC_VALID_FOLDER_PATH = DYNAMIC_TRAIN_FOLDER_PATH
    
    LIDAR_RANGE = 120
    
    # train_file  = sorted(os.listdir( DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[0]
    val_file  = sorted(os.listdir(DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[-1]
    # val_file  = sorted(os.listdir(DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[0]
    npyList = sorted(os.listdir(DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[:-1]
    print("val file is")
    print(val_file)
    print(npyList)
    # assert False
   
    def making_mask(generated_run_npy):
        dynamic_only_array=(generated_run_npy[:,2,:,:]==1)
        static_only_array=np.invert(dynamic_only_array)
        dynamic_only_array=np.array(dynamic_only_array,dtype=np.int8)
        static_only_array=np.array(static_only_array,dtype=np.int8)
        dynamic_only_array=dynamic_only_array.reshape(generated_run_npy.shape[0],1,40,512)
        static_only_array=static_only_array.reshape(generated_run_npy.shape[0],1,40,512)
        mask=np.concatenate((static_only_array,dynamic_only_array),axis=1)
        print("mask ki shape:",mask.shape)
        return mask

    print("Dynamic preprocessing of:")
    if not os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH):
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)
    else:
        shutil.rmtree(DYNAMIC_PREPROCESS_DATA_PATH)
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)
    if not os.path.exists(PATH_FOR_MASK_SAVING_AND_LOADING):
        os.makedirs(PATH_FOR_MASK_SAVING_AND_LOADING)
    else:
        shutil.rmtree(PATH_FOR_MASK_SAVING_AND_LOADING)
        os.makedirs(PATH_FOR_MASK_SAVING_AND_LOADING)

    print("validation dataset:")    
    dynamic_dataset_val   = np.load(os.path.join(DYNAMIC_VALID_FOLDER_PATH, val_file))
    print(dynamic_dataset_val.shape)
    dynamic_dataset_val   = preprocess(dynamic_dataset_val, LIDAR_RANGE).astype('float32')
    gc.collect()
    mask_val              = making_mask(dynamic_dataset_val)
    np.save(os.path.join(PATH_FOR_MASK_SAVING_AND_LOADING,val_file),mask_val)
    np.save(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file),dynamic_dataset_val[:,:2])
    del mask_val
    gc.collect()
    
#    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), 'wb') as pfile:
#        pickle.dump(dynamic_dataset_val[:,:2], pfile, protocol=pickle.HIGHEST_PROTOCOL)
    del val_file
    gc.collect()

    # raise SysError
    
    print("training dataset:")
    for file in npyList:
        print(file)
        dynamic_dataset_train = np.load(os.path.join(DYNAMIC_TRAIN_FOLDER_PATH,file))
        dynamic_dataset_train = preprocess(dynamic_dataset_train, LIDAR_RANGE).astype('float32')
        gc.collect()
        mask_train            = making_mask(dynamic_dataset_train)
        np.save(os.path.join(PATH_FOR_MASK_SAVING_AND_LOADING,file),mask_train)
        np.save(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file),dynamic_dataset_train[:,:2])
        del mask_train
        gc.collect()
    
    
#        with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file), 'wb') as pfile:
#            pickle.dump(dynamic_dataset_train[:,:2], pfile, protocol=pickle.HIGHEST_PROTOCOL)
        del dynamic_dataset_train
        gc.collect()

    assert False

else:

    # train_file  = sorted(os.listdir( DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[0]
    val_file  =   sorted(os.listdir(DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[-1]
    npyList = sorted(os.listdir(DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[:-1]
    print(npyList)
    if os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH):
        print("Have already preprocessed datasets at {}".format(DYNAMIC_PREPROCESS_DATA_PATH))
    else:
        print("No preprocessed datasets at {} and {}".format(DYNAMIC_PREPROCESS_DATA_PATH))
        assert False
    if os.path.exists(PATH_FOR_MASK_SAVING_AND_LOADING):
        print("Have already preprocessed datasets at {}".format(PATH_FOR_MASK_SAVING_AND_LOADING))
    else:
        print("No preprocessed datasets at {} and {}".format(PATH_FOR_MASK_SAVING_AND_LOADING))
        assert False
     
    print("Loading dynamic validation dataset")
#    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), 'rb') as pkl_file:
#        dynamic_dataset_val = pickle.load(pkl_file)
    dynamic_dataset_val  = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file))
    gc.collect()

    print("Loading mask for validation")
    mask_val = np.load(os.path.join(PATH_FOR_MASK_SAVING_AND_LOADING,val_file))
    gc.collect()
    val_data = Pairdata(dynamic_dataset_val, mask_val, transform = transforms.Compose([add_noise_partial()]))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=True, drop_last=False)
    gc.collect()


    print("Loading and creating training datalaoders !")
    train_loader_list = []
    for file in tqdm(npyList):
#        with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file), 'rb') as pkl_file:
#            dynamic_dataset_train = pickle.load(pkl_file)
        dynamic_dataset_train = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file))
        gc.collect()
        mask_train = np.load(os.path.join(PATH_FOR_MASK_SAVING_AND_LOADING, file))
        gc.collect()

        train_data = Pairdata(dynamic_dataset_train, mask_train, transform = transforms.Compose([add_noise_partial()]))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                    shuffle=True, drop_last=False)
        gc.collect()

        train_loader_list.append(train_loader)
        gc.collect()

   # assert False

def loss_fn(model_output,target):
    loss=nn.L1Loss()
    loss_value=loss(model_output,target.float())
    return loss_value

    # VAE training
# ----------------------- AE TRAINING -------------------------------------------------------
rangee=150 if args.autoencoder else 300
# print("Begin training:")
for epoch in range(500):
    print('Epoch #%s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x


    # FOR EVERY SMALL FILE
    print("Training: ")
    for idx, train_loader in enumerate(random.sample(train_loader_list,3)):
        print("Training on big batch {} / {}".format(idx+1, len(train_loader_list)))
    
        for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # if i == 100:
            #    break
            dynamic_img = img_data[1].cuda()
            static_img  = img_data[2].cuda()
    #         dynamic_img = dynamic_img[:,:2,:,:]

    #         print(dynamic_img.shape)
    #         assert False

            recon = model(process_input(dynamic_img))
            static_img = process_input(static_img)
            # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
            loss_recon = loss_fn(recon, torch.unsqueeze(static_img,1))
#            print(static_img)

            # if args.autoencoder:
            #     kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
            # else:
            #     kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
            #                     torch.clamp(kl_cost, min=5)

            loss = (loss_recon)
            # elbo = (kl_cost + loss_recon)

            # loss_    += [loss.item()]
            # elbo_    += [elbo.item()]
            # kl_cost_ += [kl_cost.mean(dim=0).item()]
            # kl_obj_  += [kl_obj.mean(dim=0).item()]
            recon_   += [loss_recon.mean(dim=0).item()]

            # baseline loss is very memory heavy 
            # we accumulate gradient to simulate a bigger minibatch
            # if (i+1) % factor == 0 or not is_baseline: 
            optim.zero_grad()

            loss.backward()
            # if (i+1) % factor == 0 or not is_baseline: 
            optim.step()
        
            scheduler.step()
    #####

    writes += 1
    mn = lambda x : np.mean(x)
    # print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    # print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
    # print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
    # print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
    print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
    recon_ = []
    gc.collect()
        
    # save some training reconstructions
    # if epoch % 5 == 0:
    #      recon = recon[:ns].cpu().data.numpy()
    #      with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
    #          np.save(f, recon)

         # print('saved training reconstructions')
         
    
    # Testing loop   --------------------------------------------------------------------------
    
    print("Validating: ")
#     assert False
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            # print('test set evaluation')
            for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                dynamic_img = img_data[1].cuda()
                static_img  = img_data[2].cuda()

                recon = model(process_input(dynamic_img))
                static_img = process_input(static_img)
           
                # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
                loss_recon = loss_fn(recon, torch.unsqueeze(static_img,1))

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
            gc.collect()

            # if epoch % 10 == 0:
            #     with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
            #         recon = recon[:ns].cpu().data.numpy()
            #         np.save(f, recon)
            #         # print('saved test recons')
               
            #     sample = model.sample()
            #     with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
            #         sample = sample.cpu().data.numpy()
            #         np.save(f, recon)
                
            #     # print('saved model samples')
                
            # if epoch == 0: 
            #     with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
            #         static_img = static_img.cpu().data.numpy()
            #         np.save(f, static_img)
                
                # print('saved real LiDAR')

    # assert False
    torch.save(model.state_dict(), os.path.join(args.base_dir, 'models_noise_kitti_test/gen_{}.pth'.format(epoch)))
    gc.collect()
