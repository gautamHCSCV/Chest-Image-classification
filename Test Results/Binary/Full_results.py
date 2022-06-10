import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[93]:

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics





##############################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="7"
##############################################################


# ### Use device for `cuda` or `cpu` based on availability

# In[5]:


####################################################################
#GPU using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
####################################################################


# In[6]:


config = dict(
    IMAGE_PATH= "nih-dataset/nih-15-class",
    saved_path="saved/resnet18.pt",
    lr=0.001, 
    EPOCHS = 5,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=8,
    USE_AMP = True,
    channels_last=False)
####################################################################


# In[7]:


# For custom operators, you might need to set python seed
random.seed(config['SEED'])
# If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG 
np.random.seed(config['SEED'])
# Prevent RNG for CPU and GPU using torch
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])

torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# Apply Data Transforms (Aumentations + Processing)

# In[9]:


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


print('STANFORD DATASET')


my_path = '../../../dataset/STANFORD_DATA/stanford_chest_train/'


# In[33]:


import torchvision
test_data = torchvision.datasets.ImageFolder(root=my_path,transform=data_transforms['test'])
print(len(test_data))


# In[35]:


print(test_data.class_to_idx)


test_dl = torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])



criterion = nn.CrossEntropyLoss()

def evaluation(model,test_dl):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    preds = []
    pred_labels = []
    labels = []

            # Disable gradient calculation for validation or inference using torch.no_rad()
    with torch.no_grad():
                for x,y in test_dl:
                    x = x.to(config['device'])
                    y = y.to(config['device']) #CHW --> #HWC
                    valid_logits = model(x)
                    predict_prob = F.softmax(valid_logits)
                    _,predictions = predict_prob.max(1)
                    predictions = predictions.to('cpu')

                    _, valid_preds = torch.max(valid_logits, 1)
                    valid_loss = criterion(valid_logits,y)
                    running_loss += valid_loss.item() * x.size(0)
                    running_corrects += torch.sum(valid_preds == y.data)
                    total += y.size(0)
                    predict_prob = predict_prob.to('cpu')

                    pred_labels.extend(list(predictions.numpy()))
                    preds.extend(list(predict_prob.numpy()))
                    y = y.to('cpu')
                    labels.extend(list(y.numpy()))

    epoch_loss = running_loss / len(test_data)
    epoch_acc = running_corrects.double() / len(test_data)
    print("Test Loss is {}".format(epoch_loss))
    print("Test Accuracy is {}".format(epoch_acc.cpu()))
    return np.array(labels),np.array(pred_labels),np.array(preds)






# In[105]:


mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 2, bias = True)
model = mobilenet
optimizer = optim.Adam(model.parameters(),lr=config['lr'])


# In[106]:


model.load_state_dict(torch.load('saved/mobilenetv2.pt'))

model = model.to(config['device'])
# In[112]:
print('\nmobilenet-v2')

labels, pred_labels, preds = evaluation(model,test_dl)
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))
print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[115]:


pred_mobv2_1 = copy.deepcopy(np.array(preds[:,1]))
y_test_mobv2_1 = np.array(labels)

# ## Mobilenet v3

# In[116]:


mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 2, bias = True)
model = mobilenet
model.load_state_dict(torch.load('saved/mobilenetv3.pt'))


# In[117]:


model = model.to(config['device'])


# In[118]:

print('\nmobilenet-v3')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))
print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())

print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[121]:


pred_mobv3_1 = copy.deepcopy(np.array(preds[:,1]))
y_test_mobv3_1 = np.array(labels)

# Shufflenet
print('\nshufflenet\n')
shufflenet = models.shufflenet_v2_x1_0(pretrained = True)
shufflenet.fc = nn.Linear(in_features = 1024, out_features = 2, bias=True)
model = shufflenet
model.load_state_dict(torch.load('saved/shufflenetv1.pth'))
model = model.to(config['device'])
labels, pred_labels, preds = evaluation(model,test_dl)


print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))
print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))
pred_shuff_1 = np.array(preds[:,1])
y_test_shuff_1 = np.array(labels)


# ## ExLNet

# In[146]:


squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
if 1==1:
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
            nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

model = squeezenet


# In[147]:


model.load_state_dict(torch.load('saved/exlbn1_best.pt'))
model = model.to(config['device'])


# In[148]:

print('\nExlnet\n')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[151]:


pred_exlnet_1 = copy.deepcopy(np.array(preds[:,1]))
y_test_exlnet_1 = np.array(labels)

# ## ResNet50

# In[128]:


resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features = 2048, out_features = 2, bias = True))
model = resnet50

model.load_state_dict(torch.load('saved/resnet50.pt'))
model = model.to(config['device'])


# In[129]:

print('resnet-50')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))

print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[132]:


pred_res50_1 = copy.deepcopy(np.array(preds[:,1]))
y_test_res50_1 = np.array(labels)

# ## Squeezenet

# In[133]:


squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
squeezenet.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model = squeezenet
model.load_state_dict(torch.load('saved/squeezenet.pt'))
model = model.to(config['device'])


# In[134]:

print('squeezenet')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


y_test_1 = np.array(labels)
pred_squeeze_1 = copy.deepcopy(np.array(preds[:,1]))


# VINBIG
print('\n\n VINBIG DATASET\n')

my_path = '../../../dataset/VINBIG_DATA/vinbig_2class_alldata/'


test_data = torchvision.datasets.ImageFolder(root=my_path,transform=data_transforms['test'])
print(len(test_data))


# In[35]:


print(test_data.class_to_idx)


test_dl = torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])




# In[105]:


mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 2, bias = True)
model = mobilenet

optimizer = optim.Adam(model.parameters(),lr=config['lr'])

# In[106]:


model.load_state_dict(torch.load('saved/mobilenetv2.pt'))

model = model.to(config['device'])
# In[112]:
print('\nmobilenet-v2')

labels, pred_labels, preds = evaluation(model,test_dl)
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))
print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[115]:


pred_mobv2_2 = copy.deepcopy(np.array(preds[:,1]))
y_test_mobv2_2 = np.array(labels)

# ## Mobilenet v3

# In[116]:


mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 2, bias = True)
model = mobilenet
model.load_state_dict(torch.load('saved/mobilenetv3.pt'))


# In[117]:


model = model.to(config['device'])


# In[118]:

print('mobilenet-v3')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))

print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[121]:


pred_mobv3_2 = copy.deepcopy(np.array(preds[:,1]))
y_test_mobv3_2 = np.array(labels)

# Shufflenet
print('\nshufflenet\n')
shufflenet = models.shufflenet_v2_x1_0(pretrained = True)
shufflenet.fc = nn.Linear(in_features = 1024, out_features = 2, bias=True)
model = shufflenet
model.load_state_dict(torch.load('saved/shufflenetv1.pth'))
model = model.to(config['device'])
labels, pred_labels, preds = evaluation(model,test_dl)


print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))
print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))
pred_shuff_2 = np.array(preds[:,1])
y_test_shuff_2 = np.array(labels)


# ## ExLNet

# In[146]:
squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
if 1==1:
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
            nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

model = squeezenet


# In[147]:


model.load_state_dict(torch.load('saved/exlbn1_best.pt'))
model = model.to(config['device'])


# In[148]:

print('exlnet')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))

print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[151]:


pred_exlnet_2 = copy.deepcopy(np.array(preds[:,1]))
y_test_exlnet_2 = np.array(labels)

# ## ResNet50

# In[128]:


resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features = 2048, out_features = 2, bias = True))
model = resnet50

model.load_state_dict(torch.load('saved/resnet50.pt'))
model = model.to(config['device'])


# In[129]:

print('resnet-50')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[132]:


pred_res50_2 = copy.deepcopy(np.array(preds[:,1]))
y_test_res50_2 = np.array(labels)

# ## Squeezenet

# In[133]:


squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
squeezenet.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model = squeezenet
model.load_state_dict(torch.load('saved/squeezenet.pt'))
model = model.to(config['device'])


# In[134]:

print('squeezenet')
labels, pred_labels, preds = evaluation(model,test_dl)

print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


print(metrics.classification_report(labels,pred_labels,target_names = ['abnormal','normal']))
cm = metrics.confusion_matrix(labels,pred_labels)
print('\n classwise accuracy')
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[137]:


y_test_2 = np.array(labels)
pred_squeeze_2 = copy.deepcopy(np.array(preds[:,1]))



plt.figure(figsize=(6,12))
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(6)
fig.set_figwidth(18)

fpr, tpr, threshold = metrics.roc_curve(y_test_mobv2_1, pred_mobv2_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'Mobilenet-V2: %0.2f' % roc_auc,color = 'orange')

fpr, tpr, threshold = metrics.roc_curve(y_test_shuff_1, pred_shuff_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'Shufflenet: %0.2f' % roc_auc,color = 'pink')

fpr, tpr, threshold = metrics.roc_curve(y_test_mobv3_1, pred_mobv3_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'Mobilenet-V3: %0.2f' % roc_auc,color = 'blue')

fpr, tpr, threshold = metrics.roc_curve(y_test_res50_1, pred_res50_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'ResNet-50: %0.2f' % roc_auc,color = 'green')


fpr, tpr, threshold = metrics.roc_curve(y_test_1, pred_squeeze_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'Squeezenet: %0.2f' % roc_auc,color = 'yellow')

fpr, tpr, threshold = metrics.roc_curve(y_test_exlnet_1, pred_exlnet_1)
roc_auc = metrics.auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b', label = 'ExLNet: %0.2f' % roc_auc,color = 'black')

ax1.legend(loc = 'lower right')
ax1.plot([0, 1], [0, 1],'r--')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_ylabel('True Positive Rate')
ax1.set_xlabel('False Positive Rate')

fpr, tpr, threshold = metrics.roc_curve(y_test_mobv2_2, pred_mobv2_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'Mobilenet-V2: %0.2f' % roc_auc,color = 'orange')


fpr, tpr, threshold = metrics.roc_curve(y_test_shuff_2, pred_shuff_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'Shufflenet: %0.2f' % roc_auc,color = 'pink')

fpr, tpr, threshold = metrics.roc_curve(y_test_mobv3_2, pred_mobv3_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'Mobilenet-V3: %0.2f' % roc_auc,color = 'blue')

fpr, tpr, threshold = metrics.roc_curve(y_test_res50_2, pred_res50_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'ResNet-50: %0.2f' % roc_auc,color = 'green')


fpr, tpr, threshold = metrics.roc_curve(y_test_2, pred_squeeze_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'Squeezenet: %0.2f' % roc_auc,color = 'yellow')

fpr, tpr, threshold = metrics.roc_curve(y_test_exlnet_2, pred_exlnet_2)
roc_auc = metrics.auc(fpr, tpr)
ax2.plot(fpr, tpr, 'b', label = 'ExLNet: %0.2f' % roc_auc,color = 'black')

ax2.legend(loc = 'lower right')
ax2.plot([0, 1], [0, 1],'r--')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_ylabel('True Positive Rate')
ax2.set_xlabel('False Positive Rate')

ax1.set_title('Using Chexpert')
ax2.set_title('Using Vinbig')

plt.savefig('ROC.pdf')
plt.savefig('ROC.svg',format = 'svg')
plt.show()
