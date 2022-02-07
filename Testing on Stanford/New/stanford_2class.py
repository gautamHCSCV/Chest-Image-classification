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


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import copy





##############################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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


# ### Create tensors directly on the target device
# 
# Instead of calling `torch.rand(size).cuda()` to generate a random tensor, produce the output directly on the target device: `torch.rand(size, device=torch.device('cuda'))`.
# 
# This is applicable to all functions which create new tensors and accept device argument: `torch.rand(), torch.zeros(), torch.full()` and similar.

# CUDA Convolution Benchmarking: Best Convolution Algo but may loose reproducibility if True

# In[8]:


# If researcher and wants to reproducuce: False
# If developer wants performance: True
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# Apply Data Transforms (Aumentations + Processing)

# In[9]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[32]:


my_path = '../../../dataset/stanford_chest_train/'


# In[33]:


import torchvision
images = torchvision.datasets.ImageFolder(root=my_path,transform=data_transforms['test'])
print(len(images))
train_data,valid_data,test_data = torch.utils.data.dataset.random_split(images,[500,500,222414])


# In[35]:


print(images.class_to_idx)


# In[36]:


train_dl = torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])
valid_dl = torch.utils.data.DataLoader(dataset = valid_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])
test_dl = torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])


# Optimizers:
# Gradient Descent:-
#     a. Stoicastic Gradient Descent bs = 1; 'n' number of examples. 'n / 1' number of data loader/steps for 1 Epoch
#     b. Mini-Batch Gradient Descent bs = 32; 'n' number of examples. 'n / 32' number of dataloaders/step for 1 Epoch 
#     c. Full Batch Gradient Descent bs = total_number_of_samples number of dataloader/steps = 1 for 1 Epoch

# In[37]:


len(train_dl)


# In[38]:


import matplotlib.pyplot as plt
a = iter(valid_dl)
b = next(a)
print(b[1])
plt.imshow(b[0][0][0])


# Load Model : Pretrained from torchvision model zoo or Saved model

# In[105]:


mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 2, bias = True)
model = mobilenet


# In[106]:


model.load_state_dict(torch.load('saved/mobilenetv2.pt'))


# In[107]:


#Channel Last Optimization in Model

model = model.to(config['device'])


# In[108]:


optimizer = optim.Adam(model.parameters(),lr=config['lr'])
# Loss Function
criterion = nn.CrossEntropyLoss()


# ### Training Pipeline Starts

# In[109]:


def train_model(model,criterion,optimizer,num_epochs=10):

    since = time.time()                                            
    batch_ct = 0
    example_ct = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        #Training
        model.train()
        run_corrects = 0
        for x,y in train_dl:
            x = x.to(config['device'])
            y = y.to(config['device'])
            
            optimizer.zero_grad()
            #optimizer.zero_grad(set_to_none=True)
            ######################################################################
            
            train_logits = model(x) #Input = [BS,3,224,224] (Image) -- Model --> [BS,4] (Output Scores)
            _, train_preds = torch.max(train_logits, 1)
            train_loss = criterion(train_logits,y)
            run_corrects += torch.sum(train_preds == y.data)
            
            train_loss.backward() # Backpropagation this is where your W_gradient
            loss=train_loss

            optimizer.step() # W_new = W_old - LR * W_gradient 
            example_ct += len(x) 
            batch_ct += 1
            
            ########################################################################
            # Stores Wandb Logs here
            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
            ########################################################################
        
        #validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        # Disable gradient calculation for validation or inference using torch.no_rad()
        with torch.no_grad():
            for x,y in valid_dl:
                x = x.to(config['device'])
                y = y.to(config['device']) #CHW --> #HWC
                valid_logits = model(x)
                _, valid_preds = torch.max(valid_logits, 1)
                valid_loss = criterion(valid_logits,y)
                running_loss += valid_loss.item() * x.size(0)
                running_corrects += torch.sum(valid_preds == y.data)
                total += y.size(0)
            
        epoch_loss = running_loss / len(valid_data)
        epoch_acc = running_corrects.double() / len(valid_data)
        train_acc = run_corrects.double() / len(train_data)
        print("Train Accuracy",train_acc.cpu())
        print("Validation Loss is {}".format(epoch_loss))
        print("Validation Accuracy is {}".format(epoch_acc.cpu()))

            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    torch.save(model.state_dict(), config['saved_path'])


# In[110]:


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


# In[111]:


#train_model(model, criterion, optimizer, num_epochs=2)


# In[112]:
print('mobilenet-v2')

import torch.nn.functional as F
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


# In[113]:

print('Mobilenet V2')
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


# In[114]:


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[115]:


pred_mobv2 = copy.deepcopy(np.array(preds))
y_test_mobv2 = np.array(labels)

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

print('\n Mobilenet V3\n')
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


# In[114]:


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[121]:


pred_mobv3 = copy.deepcopy(np.array(preds))
y_test_mobv3 = np.array(labels)

# ## ExLNet

# In[146]:


squeezenet = torchvision.models.squeezenet1_0(pretrained=True)

squeezenet.features[3].expand1x1 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1),groups=16)
squeezenet.features[3].expand3x3 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=16)

squeezenet.features[4].expand1x1 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1),groups=16)
squeezenet.features[4].expand3x3 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=16)

squeezenet.features[5].expand1x1 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1),groups=32)
squeezenet.features[5].expand3x3 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=32)

squeezenet.features[7].expand1x1 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1),groups=32)
squeezenet.features[7].expand3x3 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=32)

squeezenet.features[8].expand1x1 = nn.Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1),groups=48)
squeezenet.features[8].expand3x3 = nn.Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=48)

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


model.load_state_dict(torch.load('saved/exlnet4.pt'))
model = model.to(config['device'])


# In[148]:

print('exlnet')
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


print('\n ExLNet\n')
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


# In[114]:


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[151]:


pred_exlnet = copy.deepcopy(np.array(preds))
y_test_exlnet = np.array(labels)

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


# In[130]:


print('\n Resnet 50\n')
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


# In[114]:


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[132]:


pred_res50 = copy.deepcopy(np.array(preds))
y_test_res50 = np.array(labels)

# ## Squeezenet

# In[133]:


squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
squeezenet.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
model = squeezenet
model.load_state_dict(torch.load('saved/squeezenet.pt'))
model = model.to(config['device'])


# In[134]:

print('squeezenet')
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


print('/n Squeezenet')
print(precision_recall_fscore_support(np.array(labels), np.array(pred_labels), average='binary'))


# In[114]:


#y_pred = np.transpose([pred[:, 1] for pred in preds])
print(roc_auc_score(np.array(labels), np.array(preds)[:,1]))


# In[137]:


y_test = np.array(labels)
pred_squeeze = copy.deepcopy(np.array(preds))


# In[153]:


preds = pred_mobv2[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_mobv2, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'Mobilenet-V2: %0.2f' % roc_auc,color = 'red')

preds = pred_mobv3[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_mobv3, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'Mobilenet-V3: %0.2f' % roc_auc,color = 'blue')

preds = pred_res50[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_res50, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'ResNet-50: %0.2f' % roc_auc,color = 'green')

preds = pred_squeeze[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'Squeezenet: %0.2f' % roc_auc,color = 'yellow')

preds = pred_exlnet[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_exlnet, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'ExLNet: %0.2f' % roc_auc,color = 'black')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE')
plt.savefig('stanford_ROC.png')
plt.show()


# In[ ]:




