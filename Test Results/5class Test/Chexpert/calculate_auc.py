import pickle
import os
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

pickle_files = list(filter(lambda x: 'pkl' in x,os.listdir()))
print(pickle_files)


for files in pickle_files:
    model_name = files[4:-21]
    with open(files,'rb') as f:
        data = pickle.load(f)
        classes = data['preds'].shape[1]
        binary_labels = label_binarize(data['labels'],classes=[i for i in range(classes)])
        plt.figure(figsize=(5,4))
        for i in range(classes):
            fpr,tpr,_ = roc_curve(binary_labels[:,i],data['preds'][:,i])
            aucc = auc(fpr,tpr)
            plt.plot(fpr,tpr,label=f'auc{i}:{aucc:.2f}')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.title(f'roc-{model_name}')
        plt.legend()
        plt.savefig(f'roc-{model_name}.svg',format='svg')
        plt.show()