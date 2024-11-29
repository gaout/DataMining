import os
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50, swin_b, Swin_B_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, Dataset
#from torcheval.metrics.functional import binary_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data_dir = '/data/ODIR_Data'

select_model_num = 3
set_resnet = select_model_num == 0
set_transformer = select_model_num == 1
set_efficient = select_model_num == 2
set_efficient_self = select_model_num == 3
need_train = True
need_test = True

if set_resnet:
    name = 'resnet'
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
if set_transformer:
    name = 'transformer'
    model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, 1)
if set_efficient:
    name = 'efficient'
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 1)
    print(model.classifier)
if set_efficient_self:
    name = 'efficient_self'
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 1)
    print(model.classifier)

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

aug_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast()
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def plotROC(fpr, tpr, roc_auc, name):
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr[0], tpr[0], color='red', lw=2, label=f'Overall ROC curve (area = {roc_auc[0]:.2f})')
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=2, label=f'Male ROC curve (area = {roc_auc[1]:.2f})')
    plt.plot(fpr[2], tpr[2], color='green', lw=2, label=f'Female ROC curve (area = {roc_auc[2]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(name + '.png')

class NPZDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = np.load(file_path)
        # Assuming your .npz files contain 'features' and 'labels' arrays
        features = transform(data['slo_fundus'])
        aug_features = aug_transform(data['slo_fundus'])
        labels = data['dr_class']
        gender = data['male']
        return torch.tensor(features, dtype=torch.float32), torch.tensor(aug_features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(gender, dtype=torch.int32)

train_dataset = NPZDataset(data_dir, split='train')
val_dataset = NPZDataset(data_dir, split='val')
test_dataset = NPZDataset(data_dir, split='test')

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

valloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.8):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z_i.device)
        positives = similarity_matrix[mask].view(batch_size, 2)
        negatives = similarity_matrix[~mask].view(batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)
        logits = logits / self.temperature
        loss = nn.functional.cross_entropy(logits, labels)
        return loss

# Model setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("Using CPU.")
model = model.to(device)

if need_train:
    # Optimizer and loss function
    criterion = nn.BCELoss()
    criterion2 = ContrastiveLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Training loop
    num_epochs = 50
    best_val_loss = torch.finfo(torch.float32).max

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, aug_inputs, labels, genders in trainloader:
            inputs, aug_inputs, labels = inputs.to(device), aug_inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            if set_efficient_self:
                aug_outputs = model(aug_inputs)
                aug_outputs = torch.nn.functional.sigmoid(aug_outputs)
                loss = 0.5 * loss + 0.5 * criterion2(outputs, aug_outputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        print(f"Train Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
        correct = 0.0
        total = 0.0
        running_loss = 0.0
        for inputs, aug_inputs, labels, genders in valloader:
            inputs, aug_inputs, labels = inputs.to(device), aug_inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            if set_efficient_self:
                aug_outputs = model(aug_inputs)
                aug_outputs = torch.nn.functional.sigmoid(aug_outputs)
                loss = 0.5 * loss + 0.5 * criterion2(outputs, aug_outputs)
            predicted = (outputs > 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            running_loss += loss.item()
        accuracy = 100 * correct / total
        print(f"Val Epoch {epoch+1}, Loss: {running_loss/len(valloader)}, Accuracy: {accuracy}")

        # Validation step (simulated)
        val_loss = running_loss / len(valloader)  # Replace with actual validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model's state dictionary
            torch.save(model.state_dict(), data_dir + '/'+ name +'_best.pth')
        scheduler.step(val_loss)

if need_test:
    # Evaluation
    # Load the model's state dictionary
    model.load_state_dict(torch.load(data_dir + '/'+ name +'_best.pth'))
    model = model.cpu()
    model.eval()
    correct = 0.0
    total = 0.0
    all_outputs, all_labels = [], []
    male_outputs, male_labels = [], []
    female_outputs, female_labels = [], []
    with torch.no_grad():
        for inputs, _, labels, genders in testloader:
            #inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)

            outputs = outputs.squeeze(1)
            predicted = (outputs > 0.5).int()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            is_male = genders == 1
            is_female = genders != 1
            all_outputs.append(outputs)
            all_labels.append(labels)
            male_outputs.append(torch.masked_select(outputs, is_male))
            male_labels.append(torch.masked_select(labels, is_male))
            female_outputs.append(torch.masked_select(outputs, is_female))
            female_labels.append(torch.masked_select(labels, is_female))

            # Compute binary confusion matrix
            #cm = binary_confusion_matrix(outputs, labels.to(torch.int64))
            #print(cm)

    accuracy = 100 * correct / total

    print(f'Accuracy on the classification test images: {accuracy:.3f}%')

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    male_outputs = torch.cat(male_outputs)
    male_labels = torch.cat(male_labels)
    female_outputs = torch.cat(female_outputs)
    female_labels = torch.cat(female_labels)
    
    fpr, tpr, roc_auc = [0.,0.,0.], [0.,0.,0.], [0.,0.,0.]

    roc_auc[0] = roc_auc_score(all_labels.int().numpy(), all_outputs.numpy())
    fpr[0], tpr[0], _ = roc_curve(all_labels.int().numpy(), all_outputs.numpy())
    print(f'AUC overall is {roc_auc[0]:.3f}')

    roc_auc[1] = roc_auc_score(male_labels.int().numpy(), male_outputs.numpy())
    fpr[1], tpr[1], _ = roc_curve(male_labels.int().numpy(), male_outputs.numpy())
    print(f'AUC male is {roc_auc[1]:.3f}')

    roc_auc[2] = roc_auc_score(female_labels.int().numpy(), female_outputs.numpy())
    fpr[2], tpr[2], _ = roc_curve(female_labels.int().numpy(), female_outputs.numpy())
    print(f'AUC female is {roc_auc[2]:.3f}')
    
    plotROC(fpr, tpr, roc_auc, name + '_roc')