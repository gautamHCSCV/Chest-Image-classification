import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support,classification_report,roc_auc_score
import torchvision.transforms as transforms
import torchvision
gpu_id = 3
device = f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu"

with open('../../../dataset/VINBIG_DATA/test_vinbig.pkl','rb') as f:
    test_data = pkl.load(f)
    
    
transformations = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def evaluate(model,loader):
    model.eval()
    correct = 0
    pred_prob = []
    y_true = []
    y_pred = []
    label_map = dict(zip(list(range(6)),[0]*6))
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_data):
            data = transformations(data)
            data = data.to(device)
            scores = model(data)
            scores_prob = torch.nn.functional.softmax(scores,dim=1)
            _,predicted = scores_prob.max(dim=1)
            predicted = predicted.cpu().item()
            flag = 0
            for t in target:
                tt = t.item()
                if predicted == tt:
                    flag = 1
                    label_map[tt] += 1
                    y_true.append(tt)
                    y_pred.append(predicted)
                    pred_prob.extend(scores_prob.cpu().numpy())
            if (flag == 0):
                min_label = math.inf
                jdx = 0;
                for t in target:
                    tt = t.item()
                    if (label_map[tt] < min_label):
                        min_label = label_map[tt]
                        jdx = tt
                        
                label_map[jdx] += 1
                y_true.append(jdx)
                y_pred.append(predicted)
                pred_prob.extend(scores_prob.cpu().numpy())
            # print(f'batch: {idx}',end='\r')
            # print(label_map)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred_prob=np.array(pred_prob)
    # print(y_true,y_pred)
    # print(y_true == y_pred)
    
    correct = (y_true == y_pred).sum()
    accuracy = correct/len(y_true)
    model.train()
    return accuracy, y_true,y_pred,pred_prob
        

print('mobilent v2')
torchmodel = torchvision.models.mobilenet_v2(pretrained = True)
torchmodel.classifier[1] = nn.Linear(in_features=1280, out_features=6, bias=True)
# print(torchmodel)
torchmodel = torchmodel.to(device)
checkpoint = torch.load('../saved-models/mobilenetv2_6.pth')   
model = checkpoint['model']
torchmodel.load_state_dict(model)
accuracy,y_true,y_pred,pred_prob = evaluate(torchmodel,test_data)
# print(pred_prob)
precision,recall,f_score,_ = precision_recall_fscore_support(y_true,y_pred)
# print(np.unique(y_true,return_counts=True))
# print(pred_prob.shape)
print('accuracy: ',accuracy)
print('precision: ',precision)
print('recall ',recall)
print('fscore: ',f_score)
print('auc: ',roc_auc_score(y_true, pred_prob, multi_class='ovr'))
print(classification_report(y_true,y_pred))

print('shufflenet')
torchmodel = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
torchmodel.fc = nn.Linear(in_features=1024, out_features=6, bias=True)
torchmodel = torchmodel.to(device)
checkpoint = torch.load('../saved-models/shuffle5v1.pth')
model = checkpoint['model']
torchmodel.load_state_dict(model)
accuracy,y_true,y_pred,pred_prob = evaluate(torchmodel,test_data)
precision,recall,f_score,_ = precision_recall_fscore_support(y_true,y_pred)
print('accuracy: ',accuracy)
print('precision: ',precision)
print('recall ',recall)
print('fscore: ',f_score)
print('auc: ',roc_auc_score(y_true, pred_prob, multi_class='ovr'))
print(classification_report(y_true,y_pred))

# print(accuracy)
# print(precision)
# print(recall)
# print(f_score)
# print(roc_auc_score(y_true, pred_prob, multi_class='ovr'))
# print(classification_report(y_true,y_pred))
