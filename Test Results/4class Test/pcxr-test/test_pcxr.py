import torch
import torchvision
import pandas as pd
import numpy as np
import pydicom
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import random
import time
import copy
import sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class Config():
    base_path = "/DATA/dataset/pediatric_dataset/physionet.org/files/vindr-pcxr/1.0.0/"
    train_path = base_path + "train/"
    test_path = base_path + "test/"
    train_csv_path = base_path + "annotations_train.csv"
    test_csv_path = base_path + "annotations_test.csv"
    gpu_id = sys.argv[2]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    model_name = sys.argv[1]
    image_size = (224,224)
    return_logs = False
    BATCH_SIZE = 32
    pin_memory = True
    num_workers = 4
    SEED= 42
    roc_title = f'roc_{model_name}'
    saved_path = f'../saved_models/{model_name}_v1_nih.pt'
    roc_path = f'../loss_acc_roc/roc-{model_name}_pcxr_method2.svg'
    fta_path = f'../roc_pickle_files/fta_{model_name}_pcxr_method2.pkl'
    

config = Config()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
print(config.device)
print(f'model: {config.model_name}')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def progress(current,total):
    progress_percent = (current * 30 / total)
    progress_percent_int = int(progress_percent)
    print(f"|{chr(9608)* progress_percent_int}{' '*(30-progress_percent_int)}|{current}/{total}",end='\r')

## ------------------------------------------- data loader (pcxr) ---------------------------##

class PCXRDataset():
    """
    ['Anterior mediastinal mass' 'Atelectasis' 'Boot-shaped heart'
    'Bronchial thickening' 'Cardiomegaly' 'Clavicle fracture' 'Consolidation'
    'Dextro cardia' 'Diffuse aveolar opacity' 'Egg on string sign'
    'Enlarged PA' 'Infiltration' 'Interstitial lung disease - ILD'
    'Lung cavity' 'Lung cyst' 'Lung hyperinflation' 'No finding'
    'Other lesion' 'Other nodule/mass' 'Other opacity'
    'Peribronchovascular interstitial opacity' 'Pleural effusion'
    'Reticulonodular opacity' 'Stomach on the right side']

    ['image_id', 'rad_ID', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max',
       'class_id']
    """
    def __init__(self,csv_file,dir_path):
        self.dir_path = dir_path
        self.class_idx = {'Atelectasis':0,
                          'Cardiomegaly':1,
                          'Consolidation':2,
                          'Pleural effusion':3}
        self.csv_ = pd.read_csv(csv_file)
        self.csv_ = self.csv_.loc[self.csv_['class_name'].isin(self.class_idx),['image_id','class_name']]
        self.csv_ = self.csv_.drop_duplicates(subset=['image_id'])
        self.csv_['class_name'] = self.csv_['class_name'].apply(lambda x:self.class_idx[x])
        self.images = np.array(self.csv_)

        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.image_size),
            torchvision.transforms.ToTensor()
        ])
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_id, label = self.images[idx]
        img_path = os.path.join(self.dir_path,f'{img_id}.dicom')
        ds = pydicom.dcmread(img_path)
        new_img = ds.pixel_array.astype('float')
        new_img = np.maximum(new_img,0) / new_img.max()
        new_img = (new_img * 255).astype(np.uint8)
        final_img = Image.fromarray(new_img)
        final_img = final_img.convert('RGB')
        final_img = self.transformations(final_img)
        return final_img, label

def GetDataloader():
    test_data = PCXRDataset(config.train_csv_path,config.train_path)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory = config.pin_memory,
        num_workers = config.num_workers
        )
    return test_loader

# test_data = PCXRDataset(config.train_csv_path,config.train_path)
# exit(0)

## ----------------------------------------- defining cnn architecture ---------------------- ###
        
CNN_arch = None

if config.model_name == 'exlnet':
    print('defining exlnet')
    squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet.features[3].expand1x1 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1),groups=16)
    squeezenet.features[3].expand3x3 = nn.Sequential(
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=16)
        )

    squeezenet.features[4].expand1x1 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1),groups=16)
    squeezenet.features[4].expand3x3 = nn.Sequential(nn.BatchNorm2d(16),
                                                    nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=16)
                                                    )
    squeezenet.features[5].expand1x1 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1),groups=32)
    squeezenet.features[5].expand3x3 = nn.Sequential(nn.BatchNorm2d(32),
                                                    nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=32))

    squeezenet.features[7].expand1x1 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1),groups=32)
    squeezenet.features[7].expand3x3 = nn.Sequential(nn.BatchNorm2d(32),
                                                    nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=32))

    squeezenet.features[8].expand1x1 = nn.Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1),groups=48)
    squeezenet.features[8].expand3x3 = nn.Sequential(nn.BatchNorm2d(48),
                                                    nn.Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=48))

    squeezenet.features[9].expand1x1 = nn.Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1),groups=48)
    squeezenet.features[9].expand3x3 = nn.Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=48)

    squeezenet.features[10].expand1x1 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1),groups=64)
    squeezenet.features[10].expand3x3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=64)

    squeezenet.features[12].expand1x1 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1),groups=64)
    squeezenet.features[12].expand3x3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=64)

    squeezenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

    CNN_arch = squeezenet

elif config.model_name == "efficientnet":
    print('defining efficientnet')
    efficientnet = torchvision.models.efficientnet_b4(pretrained = True)
    efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = 4, bias = True)
    CNN_arch = efficientnet

elif config.model_name == "mobilenetv2":
    print('defining mobilenetv2')
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 4, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "mobilenetv3":
    print('defining mobilenetv3')
    mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 4, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "resnet":
    print('defining resnet')
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features = 2048, out_features = 4, bias = True))
    CNN_arch = resnet50

elif config.model_name == "shufflenet":
    print('defining shufflenet')
    shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
    shufflenet.fc = nn.Linear(in_features = 1024, out_features = 4, bias=True)
    CNN_arch = shufflenet

elif config.model_name == "squeezenet":
    print('defining squeezenet')
    squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
    CNN_arch = squeezenet

assert CNN_arch != None, "please select a valid model_name in config class"

## --- test _scripts------------------------------------------------------------------###

transformations = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def getparams(model):
    total_parameters = 0
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()
    print(f"total_trainable_parameters are : {round(total_parameters/1e6,2)}M")


def evaluate(model,loader):
    model.eval()
    correct = 0;samples =0
    fpr_tpr_auc = {}
    pre_prob = []
    lab = []
    predicted_labels = []

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = transformations(x)
            x = x.to(config.device)
            y = y.to(config.device)
            # model = model.to(config.device)

            scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            predictions = predictions.to('cpu')
            y = y.to('cpu')
            predict_prob = predict_prob.to('cpu')

            predicted_labels.extend(list(predictions.numpy()))
            pre_prob.extend(list(predict_prob.numpy()))
            lab.extend(list(y.numpy()))

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if config.return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        
        print('correct are {:.3f}'.format(correct/samples))

    lab = np.array(lab)
    predicted_labels = np.array(predicted_labels)
    pre_prob = np.array(pre_prob)
    
    binarized_labels = label_binarize(lab,classes=[i for i in range(4)])
    for i in range(4):
        fpr,tpr,_ = roc_curve(binarized_labels[:,i],pre_prob[:,i])
        aucc = auc(fpr,tpr)
        fpr_tpr_auc[i] = [fpr,tpr,aucc]
    
    model.train()
    with open(config.fta_path,'wb') as f:
        pickle.dump(fpr_tpr_auc,f)
    return fpr_tpr_auc,lab,predicted_labels,pre_prob


def roc_plot(fta):
    plt.figure(figsize=(5,4))
    for i in range(4):
        fpr,tpr,aucc = fta[i]
        plt.plot(fpr,tpr,label=f'auc_{i}: {aucc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(config.roc_title)
    plt.legend()
    plt.savefig(config.roc_path,format='svg')
    
## ------------------------------- test pipeline ------------------------------------------------ ###

CNN_arch = CNN_arch.to(config.device)

test_loader= GetDataloader()
print(len(test_loader))

print('=> loading checkpoint')
CNN_arch.load_state_dict(torch.load(config.saved_path))
CNN_arch.eval()
for params in CNN_arch.parameters():
    params.requires_grad = False

getparams(CNN_arch)

test_fta,y_true,y_pred,prob = evaluate(CNN_arch,test_loader)
roc_plot(test_fta)

print(classification_report(y_true,y_pred))
test_pre,test_rec,test_f1,_ = precision_recall_fscore_support(y_true,y_pred)

print('class-wise')
print(test_pre)
print(test_rec)
print(test_f1)

print('avg-out')
print(test_pre.mean())
print(test_rec.mean())
print(test_f1.mean())

