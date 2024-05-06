import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(2024)
np.random.seed(2024)

if args.server == "grace":
    os.environ['http_proxy'] = '10.73.132.63:8080'
    os.environ['https_proxy'] = '10.73.132.63:8080'
elif args.server == "faster":
    os.environ['http_proxy'] = '10.72.8.25:8080'
    os.environ['https_proxy'] = '10.72.8.25:8080'

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        if self.trans == None:
            return (self.x[idx], self.y[idx], idx)
        else:
            return (self.trans(self.x[idx]), self.y[idx])  

def main_worker():

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    import medmnist 
    from medmnist import INFO, Evaluator
    from libauc.sampler import DualSampler
    root = '/content/drive/Shareddrives/CSCE421_Final_Project/code/data'
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:, args.task_index]
    
    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0

    test_data = test_data/255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels) 

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)

    if 1 != args.eval_only:
        train_dataset = DataClass(split='train', download=True, root=root)

        train_data = train_dataset.imgs 
        train_labels = train_dataset.labels[:, args.task_index]
    
        train_labels[train_labels != args.pos_class] = 99
        train_labels[train_labels == args.pos_class] = 1
        train_labels[train_labels == 99] = 0

        train_data = train_data/255.0
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels) 

        # # implement a balanced sampler
        # train_dataset = dataset(train_data, train_labels, trans=train_transform)
        # sampler = DualSampler(train_data, batch_size=32, sampling_rate=0.5, labels=train_labels)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=False, num_workers=0, sampler=sampler)
 
    from libauc.models import resnet18 as ResNet18
    from libauc.losses import AUCMLoss
    from torch.nn import BCELoss 
    from torch.optim import SGD
    from libauc.optimizers import PESG 
    from sklearn.model_selection import KFold
    net = ResNet18(pretrained=False) 
    net = net.cuda()  
    
    lambdas = [0, .001, .025, .05]
    kfold = KFold(n_splits=5, shuffle=True)

    loss_fn = AUCMLoss()
    model_aucs = []
    for l in lambdas:
        auc_scores = []
        for train_index, val_index in kfold.split(train_data):
            train_fold, val_fold = train_data[train_index], train_data[val_index]
            train_labels_fold, val_labels_fold = train_labels[train_index], train_labels[val_index]
            
            train_dataset = dataset(train_fold, train_labels_fold, trans=train_transform)
            val_dataset = dataset(val_fold, val_labels_fold, trans=train_transform)
            
            sampler = DualSampler(train_fold, batch_size=32, sampling_rate=0.5, labels=train_labels_fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=False, num_workers=0, sampler=sampler)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batchsize, shuffle=False, num_workers=0)
            
            optimizer = PESG(net.parameters(), loss_fn=loss_fn, lr=args.lr, margin=args.margin, weight_decay=l)
            auc = train(net, train_loader, val_loader, loss_fn, optimizer, epochs=args.epochs)
            auc_scores.append(auc)
    
        model_aucs.append(np.mean(auc_scores))
        print(f'{args.data} -- Lambda {l}: AUC {np.mean(auc_scores)}')
        
    best_lambda = lambdas[np.argmax(model_aucs)]
    print(f'The best regularization term for {args.data} is {best_lambda}')

def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    max_auc = 0
    for e in range(epochs):
        net.train()
        for data, targets in train_loader:
            #print("data[0].shape: " + str(data[0].shape))
            #exit() 
            targets = targets.to(torch.float32)
            data, targets = data.cuda(), targets.cuda()
            logits = net(data)
            preds = torch.flatten(torch.sigmoid(logits))
            #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
            #print("preds:" + str(preds), flush=True)
            #print("targets:" + str(targets), flush=True)
            loss = loss_fn(preds, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_auc = evaluate(net, test_loader, epoch=e)
        if epoch_auc > max_auc:
            max_auc = epoch_auc
            # os.makedirs("saved_model", exist_ok=True)
            # torch.save(net.state_dict(), f"saved_model/{args.data}-v2.1")
    #print("Max AUC: " + str(max_auc))
    print("Finished Training Instance")
    return max_auc 
  
def evaluate(net, test_loader, epoch=-1):
    # Testing AUC
    net.eval() 
    score_list = list()
    label_list = list()
    for data, targets in test_loader:
        data, targets = data.cuda(), targets.cuda()
                
        score = net(data).detach().clone().cpu()
        score_list.append(score)
        label_list.append(targets.cpu()) 
    test_label = torch.cat(label_list)
    test_score = torch.cat(score_list)
                   
    test_auc = metrics.roc_auc_score(test_label, test_score)                   
    #print("Epoch:" + str(epoch) + "Test AUC: " + str(test_auc), flush=True)
    return test_auc
     
if __name__ == "__main__":
    main_worker()
