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
    from torch.utils.data import WeightedRandomSampler
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

        train_dataset = dataset(train_data, train_labels, trans=train_transform)

        #trying to balance the dataset
        class_weights = [3, 8]
        sample_weights = [0] * len(train_dataset)
        for i, (data, label) in enumerate(train_dataset):
            class_weight = class_weights[label]
            sample_weights[i] = class_weight
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)        

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=False, num_workers=0, sampler=sampler)
 
    from libauc.models import resnet18 as ResNet18
    from libauc.losses import AUCMLoss
    from torch.nn import BCELoss 
    from torch.optim import SGD
    from libauc.optimizers import PESG 
    net = ResNet18(pretrained=False) 
    net = net.cuda()  

    weight_decay_value = .1
    
    if args.loss == "CrossEntropy" or args.loss == "CE" or args.loss == "BCE":
        loss_fn = BCELoss() 
        optimizer = SGD(net.parameters(), lr=0.1)
    elif args.loss == "AUCM":
        loss_fn = AUCMLoss()
        optimizer = PESG(net.parameters(), loss_fn=loss_fn, lr=args.lr, margin=args.margin, weight_decay=weight_decay_value)
     
    if 1 != args.eval_only:
        train(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
    
    # to save a checkpoint in training: torch.save(net.state_dict(), "saved_model/test_model") 
    if 1 == args.eval_only: 
        net.load_state_dict(torch.load(args.saved_model_path)) 
        evaluate(net, test_loader) 

def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    epoch_aucs = []
    max_auc = 0
    for e in range(epochs):
        net.train()
        positive_class_count = 0
        negative_class_count = 0
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
            #positive_class_count += (targets == 1).sum().item()
            #negative_class_count += (targets == 0).sum().item()

        #print(f"Positive/Negative Class Count: {positive_class_count}/{negative_class_count}")
        epoch_auc = evaluate(net, test_loader, epoch=e)
        if epoch_auc > max_auc:
            max_auc = epoch_auc
            os.makedirs("saved_model", exist_ok=True)
            torch.save(net.state_dict(), f"saved_model/{args.data}-v2")
        epoch_aucs.append(epoch_auc)
    print("Max AUC: " + str(max_auc))
    return epoch_aucs 
  
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
    print("Epoch:" + str(epoch) + "Test AUC: " + str(test_auc), flush=True)
    return test_auc
     
if __name__ == "__main__":
    main_worker()