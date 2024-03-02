import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import MobileNet_V2_Weights
from torch.optim.lr_scheduler import StepLR

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root='/home/user-1/underwater_spectral_features', transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    num_classes = len(dataset.classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    weights=MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights = weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_acc_mobilenet, test_acc_mobilenet = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, device, num_epochs=25)
    test_model(model, test_loader, device, dataset.classes)
    plot_accuracies(train_acc_mobilenet, test_acc_mobilenet)

def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, device, num_epochs=25):
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()  
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)
        
        epoch_train_acc = running_corrects.double() / total_train
        train_acc_history.append(epoch_train_acc.cpu().numpy())
        
        model.eval()  
        running_corrects = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
                total_test += labels.size(0)
                
        epoch_test_acc = running_corrects.double() / total_test
        test_acc_history.append(epoch_test_acc.cpu().numpy())
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {epoch_train_acc:.4f}, Test Acc: {epoch_test_acc:.4f}')
        
        scheduler.step()
    
    return train_acc_history, test_acc_history

def test_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
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

def plot_accuracies(train_acc, test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracies')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()