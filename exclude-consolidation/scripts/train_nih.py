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
import time,sys
import copy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support

# https://stackoverflow.com/questions/62159709/pydicom-read-file-is-only-working-with-some-dicom-images

class Config():
    train_path_nih = "/DATA/dataset/NIH_WHOLE_DATASET/nih-dataset/"
    train_nih = "/DATA/dataset/NIH_WHOLE_DATASET/nih-dataset/train_val_list.txt"
    test_nih = "/DATA/dataset/NIH_WHOLE_DATASET/nih-dataset/test_list.txt"
    csv_path = "/DATA/dataset/NIH_WHOLE_DATASET/nih-dataset/Data_Entry_2017.csv"
    image_size = (224,224)
    BATCH_SIZE = 32
    pin_memory = True
    num_workers = 4
    n_class = 3
    lr=3e-4
    EPOCHS=30
    gpu_id=sys.argv[2]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    val_split = 0.1
    SEED=42
    return_logs=False
    load = False
    model_name = sys.argv[1]
    roc_title = f'roc_{model_name}'
    checkpoint = f"../saved_models/{model_name}_checkpoint_nih.pt"
    saved_path = f'../saved_models/{model_name}_v1_nih.pt'
    loss_acc_path = f'../loss_acc_roc/loss-acc-{model_name}_nih.svg'
    roc_path = f'../loss_acc_roc/roc-{model_name}_nih.svg'
    fta_path = f'../roc_pickle_files/fta_{model_name}_nih.pkl'

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
    print(f"|{chr(9608)* progress_percent_int}{'-'*(30-progress_percent_int)}|{current}/{total}",end='\r')


class NIH_data_train():
    """
    ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
       'Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width',
       'Height]
       
    ['Pneumonia', 'Cardiomegaly', 'Effusion', 'Infiltration', 'No Finding', 'Pleural_Thickening', 'Emphysema', 'Fibrosis', 'Mass', 'Edema', 'Hernia', 'Pneumothorax', 'Atelectasis', 'Nodule', 'Consolidation']
    """
    def __init__(self,csv_path,images_path,txt_path,name):
        self.name = name
        self.class_idx = {'Atelectasis':0,
                          'Cardiomegaly':1,
                          'Effusion':2}
        
        with open(txt_path,'r') as f:        
            self.images = f.read().split('\n')
        
        self.images_map = {} # maps image:image_path (all images names does not overlap)
        images_folders = list(filter(lambda x:x[:6] == 'images', os.listdir(images_path)))
        for folder in images_folders:
            img_path = os.path.join(images_path,folder,'images')
            temp_map = {i:os.path.join(img_path,i) for i in os.listdir(img_path)}
            self.images_map = {**self.images_map,**temp_map}
            
        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.image_size),
            torchvision.transforms.ToTensor()
        ])
        
        self.file_name = f'{self.name}_labels_nih.pkl'
        if os.path.exists(self.file_name):
            with open(self.file_name,'rb') as f:
                print(f'{self.name}_labels_nih file already exits')
                self.images = pickle.load(f)
        else:
            print(f'creating {self.name}_labels_nih') 
            self.image_labels = []
            self.csv_ = pd.read_csv(csv_path)
            self._images_labels()
        
    def __getitem__(self,idx):
        image_name,label = self.images[idx]
        img_path = self.images_map[image_name]
        
        img = Image.open(img_path).convert("RGB")
        img = self.transformations(img)
        return img,label
                
    def __len__(self):
        return len(self.images)
    
    def _images_labels(self):
        for idx,image in enumerate(self.images):
            image_from_csv = self.csv_.loc[self.csv_['Image Index'] == image,'Finding Labels'].values[0].split('|')
            valid_labels = list(filter(lambda x: x in self.class_idx, image_from_csv))
            if valid_labels:
                for label in valid_labels:
                    self.image_labels.append((image,self.class_idx[label]))
            if config.return_logs:
                progress(idx+1,len(self.images))
        with open(self.file_name,'wb') as f:
            pickle.dump(self.image_labels,f)
        self.images = self.image_labels
        
class NIH_data_test():
    """
    ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
       'Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width',
       'Height]
       
    ['Pneumonia', 'Cardiomegaly', 'Effusion', 'Infiltration', 'No Finding', 'Pleural_Thickening', 'Emphysema', 'Fibrosis', 'Mass', 'Edema', 'Hernia', 'Pneumothorax', 'Atelectasis', 'Nodule', 'Consolidation']
    """
    def __init__(self,csv_path,images_path,txt_path,name):
        self.name = name
        self.class_idx = {'Atelectasis':0,
                          'Cardiomegaly':1,
                          'Effusion':2}
        
        with open(txt_path,'r') as f:        
            self.images = f.read().split('\n')
        
        self.images_map = {} # maps image:image_path (all images names does not overlap)
        images_folders = list(filter(lambda x:x[:6] == 'images', os.listdir(images_path)))
        for folder in images_folders:
            img_path = os.path.join(images_path,folder,'images')
            temp_map = {i:os.path.join(img_path,i) for i in os.listdir(img_path)}
            self.images_map = {**self.images_map,**temp_map}
            
        self.transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.image_size),
            torchvision.transforms.ToTensor()
        ])
        
        self.file_name = f'{self.name}_labels_nih.pkl'
        if os.path.exists(self.file_name):
            with open(self.file_name,'rb') as f:
                print(f'{self.name}_labels_nih file already exits')
                self.images = pickle.load(f)
        else:
            print(f'creating {self.name}_labels_nih') 
            self.image_labels = []# assigning random lables to images (for multiclass)
            self.csv_ = pd.read_csv(csv_path)
            self._images_labels()
        
    def __getitem__(self,idx):
        image_name,(random_label,*all_labels) = self.images[idx]
        img_path = self.images_map[image_name]
        
        img = Image.open(img_path).convert("RGB")
        img = self.transformations(img)
        return (img,random_label,*all_labels)
                
    def __len__(self):
        return len(self.images)
    
    def _images_labels(self):
        for idx,image in enumerate(self.images):
            image_from_csv = self.csv_.loc[self.csv_['Image Index'] == image,'Finding Labels'].values[0].split('|')
            valid_labels = list(filter(lambda x: x in self.class_idx, image_from_csv))
            if valid_labels:
                random_label = self.class_idx[random.choice(valid_labels)]
                class_list = [-1 for i in range(config.n_class)]
                for label in valid_labels:
                    jdx = self.class_idx[label]
                    class_list[jdx] = jdx
                self.image_labels.append((image,(random_label,*class_list)))
            if config.return_logs:
                progress(idx+1,len(self.images))
        with open(self.file_name,'wb') as f:
            pickle.dump(self.image_labels,f)
        self.images = self.image_labels
        
def GetDataloader():
    train_data = NIH_data_train(config.csv_path,config.train_path_nih,config.train_nih,name='train')
    test_data = NIH_data_test(config.csv_path,config.train_path_nih,config.test_nih,name='test')
    img,random_label,*class_labels = test_data[0]
    
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

    return train_loader, test_loader, val_loader

def test():
    train_loader, test_loader, val_loader = GetDataloader()
    print(len(train_loader))
    print(len(test_loader))
    print(len(val_loader))
    
# ------------------------------------------------defining model -------------------------------------------

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
            nn.Conv2d(512, config.n_class, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )

    CNN_arch = squeezenet

elif config.model_name == "efficientnet":
    print('defining efficientnet')
    efficientnet = torchvision.models.efficientnet_b4(pretrained = True)
    efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = config.n_class, bias = True)
    CNN_arch = efficientnet

elif config.model_name == "mobilenetv2":
    print('defining mobilenetv2')
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet.classifier[1] = nn.Linear(in_features = 1280, out_features = config.n_class, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "mobilenetv3":
    print('defining mobilenetv3')
    mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = config.n_class, bias = True)
    CNN_arch = mobilenet

elif config.model_name == "resnet":
    print('defining resnet')
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features = 2048, out_features = config.n_class, bias = True))
    CNN_arch = resnet50

elif config.model_name == "shufflenet":
    print('defining shufflenet')
    shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
    shufflenet.fc = nn.Linear(in_features = 1024, out_features = config.n_class, bias=True)
    CNN_arch = shufflenet

elif config.model_name == "squeezenet":
    print('defining squeezenet')
    squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet.classifier[1] = nn.Conv2d(512, config.n_class, kernel_size=(1, 1), stride=(1, 1))
    CNN_arch = squeezenet

assert CNN_arch != None, "please select a valid model_name in config class"

## -------------------------------------------------------train & test---------------------------------------

transformations = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def train(model,train_loader,val_loader,lossfunction,optimizer,grad_scaler,n_epochs=200):
    tval = {'valloss':[],'valacc':[],'trainacc':[],"trainloss":[]}
    best_val_acc = 0.6
    starttime = time.time()
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            data = transformations(data)    
            data = data.to(config.device)
            target = target.to(config.device)

            with torch.cuda.amp.autocast():
                scores = model(data)    
                loss = lossfunction(scores,target)

            cur_loss += loss.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            # optimizer.step()
        
            if config.return_logs:
                progress(idx+1,len(train_loader))
                # print('TrainBatchDone:{:d}'.format(idx),end='\r') 
  
        model.eval()

        valacc = 0;valloss = 0
        vl_len = len(val_loader)
        for idx ,(data,target) in enumerate(val_loader):
            
            data = transformations(data)
            data = data.to(config.device)
            target = target.to(config.device)
        
            correct = 0;samples=0
            with torch.no_grad():
                scores = model(data)
                loss = lossfunction(scores,target)
                scores =F.softmax(scores,dim=1)
                _,predicted = torch.max(scores,dim = 1)
                correct += (predicted == target).sum()
                samples += scores.shape[0]
                valloss += loss.item() / vl_len
                valacc += correct / (samples * vl_len)
                
            if config.return_logs:
                progress(idx+1,vl_len)
                # print('ValidnBatchDone:{:d}'.format(idx),end='\r') 


        model.train()
      
        # print(correct.get_device(),samples.get_device(),len(validate_loader).get_device())
        tval['valloss'].append(float(valloss))
        tval['valacc'].append(float(valacc))
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))

        if (float(valacc) > best_val_acc):
            print('=> saving checkpoint')
            torch.save(model.state_dict(),config.checkpoint)
            best_val_acc = float(valacc)
        
        print('epoch:[{:d}/{:d}], TrainAcc:{:.3f}, TrainLoss:{:.3f}, ValAcc:{:.3f}, ValLoss:{:.3f}'.format(epochs+1,n_epochs,curacc,cur_loss,valacc,valloss)) 

    torch.save(model.state_dict(),config.saved_path)
    time2 = time.time() - starttime
    print('done time {:.3f} hours'.format(time2/3600))
    return tval

def getparams(model):
    total_parameters = 0
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()
    print(f"total_trainable_parameters are : {round(total_parameters/1e6,2)}M")


def get_predictions(predictions,random_label,labels):
    y = []
    for idx,preds in enumerate(predictions):
        if preds in labels[idx]:
            y.append(preds)
        else:
            y.append(random_label[idx])
    return torch.tensor(y)

def evaluate(model,loader):
    model.eval()
    correct = 0;samples =0
    fpr_tpr_auc = {}
    pre_prob = []
    lab = []
    predicted_labels = []
    loader_len = len(loader)
    with torch.no_grad():
        for idx,(x,random_label,*labels) in enumerate(loader):
            labels = torch.cat([i.unsqueeze(1) for i in labels],dim=1)
            x = transformations(x)
            x = x.to(config.device)
            random_label = random_label.to(config.device)
            labels = labels.to(config.device)
            scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)
            
            y = get_predictions(predictions,random_label,labels)
            
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
                
        print('correct are {:.3f}'.format(correct/samples))

    lab = np.array(lab)
    predicted_labels = np.array(predicted_labels)
    pre_prob = np.array(pre_prob)
    
    binarized_labels = label_binarize(lab,classes=[i for i in range(config.n_class)])
    for i in range(config.n_class):
        fpr,tpr,_ = roc_curve(binarized_labels[:,i],pre_prob[:,i])
        aucc = auc(fpr,tpr)
        fpr_tpr_auc[i] = [fpr,tpr,aucc]
    
    model.train()
    with open(config.fta_path,'wb') as f:
        pickle.dump(fpr_tpr_auc,f)
    return fpr_tpr_auc,lab,predicted_labels,pre_prob


def loss_acc_curve(tval):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,config.EPOCHS+1)),tval['trainloss'],label='train-loss')
    plt.plot(list(range(1,config.EPOCHS+1)),tval['trainacc'],label='train-accuracy')
    plt.plot(list(range(1,config.EPOCHS+1)),tval['valloss'],label='val-loss')
    plt.plot(list(range(1,config.EPOCHS+1)),tval['valacc'],label='val-accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss/accuracy')
    plt.title('loss_accuracy')
    plt.legend()
    plt.savefig(config.loss_acc_path,format='svg')

def roc_plot(fta):
    plt.figure(figsize=(5,4))
    for i in range(config.n_class):
        fpr,tpr,aucc = fta[i]
        plt.plot(fpr,tpr,label=f'auc_{i}: {aucc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(config.roc_title)
    plt.legend()
    plt.savefig(config.roc_path,format='svg')
    

#---------------------------------------------train and test---------------------------------------------------------

# CNN_arch = squeezenet

CNN_arch = CNN_arch.to(config.device)

train_loader,test_loader,val_loader = GetDataloader()
print(len(train_loader))
print(len(test_loader))
print(len(val_loader))

if config.load:
    print('=> loading checkpoint')
    CNN_arch.load_state_dict(torch.load(config.saved_path))
    CNN_arch.eval()
    evaluate(CNN_arch,test_loader)
    exit(0)

grad_scaler = torch.cuda.amp.GradScaler()
lossfunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=CNN_arch.parameters(),lr=config.lr)

getparams(CNN_arch)
history = train(CNN_arch,train_loader,val_loader,lossfunction,optimizer,grad_scaler,n_epochs=config.EPOCHS)
loss_acc_curve(history)

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












# images_set = set()
# for idx in range(len(csv_)):
#     img_id = csv_.iloc[idx,0]
#     if img_id not in images_set:
#         class_names = np.array(csv_.loc[csv_.image_id == img_id,'class_name'])
#         if 'No finding' in class_names:
#             self.images.append((img_id,0))
#         else:
#             self.images.append((img_id,1))
#     images_set.add(img_id)