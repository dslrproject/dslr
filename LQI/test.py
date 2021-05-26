import argparse
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import os, shutil
from torchsummary import summary
import pickle
import gc
from utils import * 
from models import * 
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type = int, default = 0)
parser.add_argument('--path', type = str)
parser.add_argument('--dataset', type = str, default = 'carla')
parser.add_argument('--model_path', type = str, default = '')
parser.add_argument('--batch_size', type = int, default = 8)
parser.add_argument('--no_polar', type=int,   default=0, help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')


# ------------------------ARGS stuff----------------------------------------------


def getint(name):
    try:
        print(name.split('.'))
        return int(name.split('.')[0])
    except Exception as e:
        print("Error occured while trying to read {}".format(name))
    return name


class add_noise_partial():
    def __init__(self):
        self.noise = list(range(5,40,5))/100

    def __call__(self, sample):
        idx, img, mask = sample
        h,w = mask.shape
        ratio = np.random.randint(5,30)/100.0
        noise = np.random.choice(self.noise)
        numdyn = int(ratio*h*w)
        numdyn = numdyn - numdyn%3
        indices = np.random.randint(0,h*w,size = numdyn)
        inda,indb,indc = np.split(indices,3)
        mask.view(-1)[indices] = 1
        means = img.reshape((2, -1)).mean(-1)
        stds  = img.reshape((2, -1)).std(-1)
        noise_tensora = torch.zeros((numdyn//3,1)).normal_(0, noise)
        noise_tensorb = torch.zeros((numdyn//3,1)).normal_(0, noise)
        noise_tensorc = torch.zeros((numdyn,2)).normal_(0, noise)
        means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]
         # normalize data
        norm = (img - means) / (stds + 1e-9)
        # add the noise
        norm.reshape((2,-1))[0][inda] = norm.reshape((2,-1))[0][inda] + noise_tensora
        norm.reshape((2,-1))[1][indb] = norm.reshape((2,-1))[1][indb] + noise_tensorb
        norm.reshape((2,-1))[indc] = norm.reshape((2,-1))[indc] + noise_tensorc
        # unnormalize
        unnorm = norm * (stds + 1e-9) + means
        return idx, unnorm, mask

class Pairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset1):
        super(Pairdata, self).__init__()
        self.dataset1 = dataset1

    def __len__(self):
        return self.dataset1.shape[0]

    def __getitem__(self, index): 
        #stride = int(self.dataset1[index].shape[2]/512)
        img = torch.from_numpy(self.dataset1[index])#[:,5:45,::2])
        if img.shape == (2,40,256):
            img = F.interpolate(img.unsqueeze(0),((img.shape[2], 512)))
            img = img.squeeze(0)
        if img.shape == (2,64,1025):
            img = img[:,5:45,::2]
        means = img.view((2, -1)).mean(-1)
        stds  = img.view((2, -1)).std(-1)
        means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]
        # normalize data
        norm = (img - means) / (stds + 1e-9)
        norm = norm.reshape(img.shape)
        return index,norm.float()
    
    def shape(self):
        return self.dataset1[0].shape


#-------------------------Loading Model Weights----------------------------
args = parser.parse_args()
model = CNNIQAnet(args).cuda()
if args.model_path == '':
    if args.dataset == 'carla':
        model_file = 'carla.pth'
        #model_file = 'models_noise_regress/gen_15.pth'
else:
    model_file = args.model_path
print("Loading model from {}".format(model_file))

network=torch.load(model_file)
model.load_state_dict(network)
print(summary(model, input_size=(2, 40, 512)))
model.eval()


#--------------Preprocessing Dataset------------------------------

preprocessed_path = args.path + "/preprocessed"
if os.path.exists(preprocessed_path):
    print("Preprocessed exists")
 
if args.preprocess and not os.path.exists(preprocessed_path):
    LIDAR_RANGE = 120
    print(os.listdir(args.path))
    npyList = sorted(os.listdir(args.path), key=getint)
    print(npyList) 
    os.makedirs(preprocessed_path)
    print("Processing:")
    for file in tqdm(npyList):
        print(file)
        dynamic_dataset_train = np.load(os.path.join(args.path,file))
        dynamic_dataset_train = preprocess(dynamic_dataset_train, LIDAR_RANGE).astype('float32')
        gc.collect()
        np.save(os.path.join(preprocessed_path, file),dynamic_dataset_train[:,:2])
        del dynamic_dataset_train
        gc.collect()

if os.path.exists(preprocessed_path):
    print("Have already preprocessed datasets at {}".format(preprocessed_path))
else:
    print("No preprocessed datasets at {} ".format(preprocessed_path))
    print("Considering data as preprocessed")
    preprocessed_path = args.path

npyList = sorted(os.listdir(preprocessed_path), key=getint)
print(npyList)

print("Loading and creating training datalaoders !")
train_loader_list = []
for file in tqdm(npyList):
    dynamic_dataset_train = np.load(os.path.join(preprocessed_path, file))
    gc.collect()
    train_data = Pairdata(dynamic_dataset_train)
    print(train_data.shape())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                shuffle=True, drop_last=False)
    gc.collect()
    train_loader_list.append(train_loader)
    gc.collect()

process_input = from_polar if args.no_polar else lambda x : x
with torch.no_grad():
    runnoiselist = []
    for idx, train_loader in enumerate(train_loader_list):
        batchnoiselist = []
        for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            dynamic_img = img_data[1].cuda()
            recon = model(process_input(dynamic_img[:,:2,:,:]))
            recon = np.array(recon.detach().cpu())
            mse_batch = np.sum((recon))/recon.size
            batchnoiselist.append(mse_batch)
        batchnoisearr = np.array(batchnoiselist)
        print("For file {}: Noise:{}".format(npyList[idx], np.sum(batchnoisearr)/batchnoisearr.size))
        runnoiselist.append(np.sum(batchnoisearr)/batchnoisearr.size)

    print("************************")
    runmsearr = np.array(runnoiselist)
    #runaccarr = np.array(: {}".format(np.sum(runmsearr)/runmsearr.size))