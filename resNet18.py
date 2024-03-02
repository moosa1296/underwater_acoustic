from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

dataset = ImageFolder(root='/home/user-1/underwater_spectral_features', transform=transform)

sample, label = dataset[0]
print(type(sample)) 

samples_per_class = {class_idx: 0 for class_idx in range(len(dataset.classes))}
print(type(samples_per_class))
for _, class_idx in dataset.samples:
    samples_per_class[class_idx] += 1


valid_classes = [class_idx for class_idx, count in samples_per_class.items() if count > 1]

filtered_samples = [(sample, class_idx) for sample, class_idx in dataset.samples if class_idx in valid_classes] 

dataset.samples = filtered_samples
dataset.targets = [class_idx for _, class_idx in filtered_samples]

indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.targets)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

num_classes = len(dataset.classes)  
model = resnet18(weights = ResNet18_Weights.DEFAULT)  
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1) 

def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, device, num_epochs=25):
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)
            
        epoch_acc_train = running_corrects.double() / total_train
        train_acc_history.append(epoch_acc_train.cpu().numpy())
        
        model.eval()
        running_corrects = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_test += labels.size(0)
                
        epoch_acc_test = running_corrects.double() / total_test
        test_acc_history.append(epoch_acc_test.cpu().numpy())
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {epoch_acc_train:.4f}, Test Acc: {epoch_acc_test:.4f}')
        scheduler.step()
    
    return train_acc_history, test_acc_history

train_acc_resnet, test_acc_resnet = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, device, num_epochs=25)


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    return accuracy, precision, recall, f1
test_model(model, test_loader)


def plot_accuracies(train_acc, test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    plt.show()

plot_accuracies(train_acc_resnet, test_acc_resnet)
def classify_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device) 
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()

    return dataset.classes[predicted_class_index]

image_path = '/home/user-1/6102604H.png'
predicted_class = classify_image(image_path, model, transform, device)
print(f'Predicted class: {predicted_class}')