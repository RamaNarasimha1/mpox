import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm                

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent directory (cs)
    
    train_dir = os.path.join(base_dir, "DATA", "aug", "aug_train")
    val_dir = os.path.join(base_dir, "DATA", "split_data", "val")
    test_dir = os.path.join(base_dir, "DATA", "split_data", "test")

                     
    batch_size = 32
    num_workers = 2
    num_epochs = 30
    patience = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

              
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def run_training(model, model_name):
        print(f"\n==== Training {model_name} ====")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_val_acc = 0.0
        epochs_no_improve = 0
        best_model_weights = None
        best_model_path = f"{model_name}_best.pth"

        def train_one_epoch():
            model.train()
            running_loss, correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            return running_loss / total, correct / total

        def evaluate(loader):
            model.eval()
            running_loss, correct, total = 0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

                       
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch()
            val_loss, val_acc, _, _ = evaluate(val_loader)

            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_model_weights = model.state_dict()
                print(f"[{model_name}] Validation accuracy improved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[{model_name}] Early stopping.")
                    break

                         
        torch.save(best_model_weights, best_model_path)
        print(f"[{model_name}] Best model saved to {best_model_path}")

                          
        model.load_state_dict(best_model_weights)
        _, test_acc, test_preds, test_labels = evaluate(test_loader)
        print(f"[{model_name}] Test Accuracy: {test_acc:.4f}")

                
        class_names = train_dataset.classes
        report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
        cm = confusion_matrix(test_labels, test_preds)

        with open(f"{model_name}_report.txt", "w") as f:
            f.write(report)

        np.save(f"{model_name}_confusion_matrix.npy", cm)

                               
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f"{model_name}_confusion_matrix.png")
        plt.close()

                               
    ghostnet = timm.create_model('ghostnet_100', pretrained=True)
    num_ftrs = ghostnet.classifier.in_features
    ghostnet.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))
    run_training(ghostnet, "ghostnet_100")

                                 
    squeezenet = models.squeezenet1_1(pretrained=True)
    squeezenet.classifier[1] = nn.Conv2d(512, len(train_dataset.classes), kernel_size=1)
    squeezenet.num_classes = len(train_dataset.classes)
    run_training(squeezenet, "squeezenet1_1")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
