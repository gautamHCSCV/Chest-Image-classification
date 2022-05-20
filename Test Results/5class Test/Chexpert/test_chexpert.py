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
import matplotlib.image as mpimg
import pickle
import random
import time
import copy
import sys
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')

class Config():
    train_path = "../../../dataset/Dvinbigdata-chest-xray-abnormalities-detection"
    train_csv = "../../../dataset/train.csv"
    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    model_name = 'exlnet'
    image_size = (224,224)
    return_logs = True
    BATCH_SIZE = 32
    pin_memory = True
    num_workers = 4
    SEED= 42
    roc_title = f'roc_{model_name}'
    saved_path = f'saved_models/{model_name}_v1_nih.pt'
    roc_path = f'loss_acc_roc/roc-{model_name}_vinbig.svg'
    fta_path = f'roc_pickle_files/fta_{model_name}_vinbig.pkl'
    

config = Config()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
device = config.device
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
            nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

    CNN_arch = squeezenet

elif config.model_name == "efficientnet":
    print('defining efficientnet')
    efficientnet = torchvision.models.efficientnet_b4(pretrained = True)
    efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = 5, bias = True)
    CNN_arch = efficientnet

elif config.model_name == "mobilenetv2":
    print('defining mobilenetv2')
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 5, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "mobilenetv3":
    print('defining mobilenetv3')
    mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 5, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "resnet":
    print('defining resnet')
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features = 2048, out_features = 5, bias = True))
    CNN_arch = resnet50

elif config.model_name == "shufflenet":
    print('defining shufflenet')
    shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
    shufflenet.fc = nn.Linear(in_features = 1024, out_features = 5, bias=True)
    CNN_arch = shufflenet

elif config.model_name == "squeezenet":
    print('defining squeezenet')
    squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1))
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

criterion = nn.CrossEntropyLoss()


# In[3]:


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# In[4]:


path0 = '../../../dataset/Stanford_chestxray_dataset/'
path = '../../../dataset/Stanford_chestxray_dataset/CheXpert-v1.0-small/'
os.listdir(path)


# In[5]:


path1 = path + 'train.csv'
df = pd.read_csv(path1)
df.head()


# In[11]:
mapping = {'No Finding':0,'Atelectasis':1,'Cardiomegaly':2,'Consolidation':3,'Pleural Effusion':4}
#mapping = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Pleural Effusion': 3, 'No Finding': 4, 'Pneumothorax': 5}
disease_labels = {j:i for i,j in mapping.items()}

diseases = ['Consolidation','Cardiomegaly','No Finding','Pleural Effusion','Atelectasis']
print(disease_labels)



# In[10]:


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[25]:


def Evaluate(model):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    preds = []
    pred_labels = []
    labels = []
    with torch.no_grad():
        for i in range(len(df)):
            if df['Frontal/Lateral'][i]=="Lateral":
                continue
            for j in diseases:
                if df[j][i]==1.0:
                    image = pil_loader(path0+df['Path'][i])
                    x = data_transforms['test'](image)
                    x = torch.Tensor(np.expand_dims(x,axis = 0))
                    x = x.to(device)
                    valid_logits = model(x)
                    predict_prob = F.softmax(valid_logits)

                    _,predictions = predict_prob.max(1)
                    predictions = predictions.to('cpu')
                    prediction = int(predictions[0])
                    if df[disease_labels[prediction]][i] and df[disease_labels[prediction]][i]== 1:
                        running_corrects += 1
                        labels.append(prediction)
                    else:
                        labels.append(mapping[j])
                    predict_prob = predict_prob.to('cpu')

                    pred_labels.extend(list(predictions.numpy()))
                    preds.extend(list(predict_prob.numpy()))
                    total += 1
                    break
        print('Accuracy:',running_corrects/total)
        return(np.array(preds), np.array(pred_labels),np.array(labels))

    
    
CNN_arch = CNN_arch.to(config.device)

print('=> loading checkpoint')
CNN_arch.load_state_dict(torch.load(config.saved_path))
CNN_arch.eval()

getparams(CNN_arch)

preds, pred_labels,labels = Evaluate(CNN_arch)
print(metrics.precision_recall_fscore_support(np.array(labels), np.array(pred_labels)))
print('\nAUROC:')
print(metrics.roc_auc_score(np.array(labels), np.array(preds), multi_class='ovr'))
print(metrics.classification_report(labels,pred_labels))    

d = {'preds':preds,'labels':labels}
with open(f'{config.model_name}.pkl','wb') as f:
    pickle.dump(d,f)