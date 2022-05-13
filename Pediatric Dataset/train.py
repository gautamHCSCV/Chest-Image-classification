import torch
import torchvision
import pandas as pd
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support

# https://stackoverflow.com/questions/62159709/pydicom-read-file-is-only-working-with-some-dicom-images

class Config():
    base_path = "../../../dataset/pediatric_dataset/physionet.org/files/vindr-pcxr/1.0.0/"
    train_path = base_path + "train/"
    test_path = base_path + "test/"
    train_csv_path = base_path + "annotations_train.csv"
    test_csv_path = base_path + "annotations_test.csv"
    image_size = (224,224)
    BATCH_SIZE = 32
    pin_memory = True
    num_workers = 3
    lr=0.001
    EPOCHS=30
    gpu_id=3
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    val_split = 0.1
    SEED=42
    return_logs=False
    load = False
    model_name = "squeezenet"
    roc_title = f'roc_{model_name}'
    checkpoint = f"../saved_models/{model_name}_checkpoint.pt"
    saved_path = f'../saved_models/{model_name}_v1.pt'
    loss_acc_path = f'../loss_acc_roc/loss-acc-{model_name}.svg'
    roc_path = f'../loss_acc_roc/roc-{model_name}.svg'
    fta_path = f'../roc_pickle_files/fta_{model_name}.pkl'

config = Config()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
print(config.device)

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PCXRDataset():
    def __init__(self,csv_file,dir_path):
        self.dir_path = dir_path
        csv_ = pd.read_csv(csv_file)

        csv_ = csv_.drop_duplicates(subset=['image_id'])
        csv_.class_name = csv_.class_name.apply(lambda x:0 if x=='No finding' else 1)
        csv_ = np.array(csv_.loc[:,['image_id','class_name']])
        self.images = csv_

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
    train_data = PCXRDataset(config.train_csv_path,config.train_path)
    test_data = PCXRDataset(config.test_csv_path,config.test_path)
    total_len = len(train_data)
    val_len = int(config.val_split * total_len)
    train_len = total_len - val_len
    training_data,val_data = torch.utils.data.dataset.random_split(train_data,[train_len,val_len])

    train_loader = torch.utils.data.DataLoader(
        training_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory = config.pin_memory,
        num_workers = config.num_workers
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory = config.pin_memory,
        num_workers = config.num_workers
        )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory = config.pin_memory,
        num_workers = config.num_workers
        )

    return train_loader, test_loader, val_loader, training_data, test_data, val_data

def test():
    train_loader, test_loader, val_loader = GetDataloader()
    print(len(train_loader))
    print(len(test_loader))
    print(len(val_loader))

# ------------------------------------------------defining model -------------------------------------------

CNN_arch = None

if config.model_name == "efficientnet":
    efficientnet = torchvision.models.efficientnet_b4(pretrained = True)
    efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = 2, bias = True)
    CNN_arch = efficientnet

elif config.model_name == "mobilenetv2":
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = 2, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "mobilenetv3":
    mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 2, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "resnet":
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features = 2048, out_features = 2, bias = True))
    CNN_arch = resnet50

elif config.model_name == "shufflenet":
    shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
    shufflenet.fc = nn.Linear(in_features = 1024, out_features = 2, bias=True)
    CNN_arch = shufflenet

elif config.model_name == "squeezenet":
    squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    CNN_arch = squeezenet


## -------------------------------------------------------train & test---------------------------------------

transformations = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train_model(model,criterion,optimizer,config,num_epochs=10):

    since = time.time()                                            
    batch_ct = 0
    example_ct = 0
    best_acc = 0.3
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        run_corrects = 0
        #Training
        model.train()
        for x,y in train_dl: #BS=32 ([BS,3,224,224], [BS,4])            
            if config['channels_last']:
                x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(config['device'])
            y = y.to(config['device']) #CHW --> #HWC
            
            
            
            optimizer.zero_grad()
            #optimizer.zero_grad(set_to_none=True)
            ######################################################################
            
            train_logits = model(x) #Input = [BS,3,224,224] (Image) -- Model --> [BS,4] (Output Scores)
            
            _, train_preds = torch.max(train_logits, 1)
            train_loss = criterion(train_logits,y)
            train_loss = criterion(train_logits,y)
            run_corrects += torch.sum(train_preds == y.data)
            
            train_loss.backward() # Backpropagation this is where your W_gradient
            loss=train_loss

            optimizer.step() # W_new = W_old - LR * W_gradient 
            example_ct += len(x) 
            batch_ct += 1
            if ((batch_ct + 1) % 80) == 0:
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
                if config['channels_last']:
                    x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
                else:
                    x = x.to(config['device'])
                y = y.to(config['device'])
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
        print("Validation Accuracy is {}\n".format(epoch_acc.cpu()))
        if epoch_acc.cpu()>best_acc:
            print('One of the best validation accuracy found.\n')
            torch.save(model.state_dict(), config['best_saved_path'])
            best_acc = epoch_acc.cpu()

            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    torch.save(model.state_dict(), config['saved_path'])

    
def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
    
config1 = dict(
    saved_path="Child/squeezenet.pt",
    best_saved_path = "Child/squeezenet_best.pt",
    lr=0.001, 
    EPOCHS = 30,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    device=config.device,
    SEED = 42,
    pin_memory=True,
    num_workers=2,
    USE_AMP = True,
    channels_last=False)
CNN_arch = CNN_arch.to(config.device)

train_dl,test_dl,valid_dl, train_data,test_data, valid_data = GetDataloader()
optimizer = optim.Adam(CNN_arch.parameters(),lr=config1['lr'])
criterion = nn.CrossEntropyLoss()
train_model(CNN_arch, criterion, optimizer,config1, num_epochs=config1['EPOCHS'])

