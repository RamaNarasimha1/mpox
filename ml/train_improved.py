import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random

# make training reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# mixup - blends images together during training
# paper showed it improves generalization - nst
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# label smoothing - prevents model from being overconfident
# rama added this after reading some papers
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # convert to one-hot
        target_one_hot = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
        
        # smooth the labels a bit
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = (-target_smooth * log_preds).sum(dim=-1).mean()
        return loss

def get_model(model_name, num_classes, pretrained=True):
    # load pretrained model and swap the head
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    train_dir = "/Users/n5t/Downloads/cs/DATA/aug/aug_train"
    val_dir = "/Users/n5t/Downloads/cs/DATA/split_data/val"
    test_dir = "/Users/n5t/Downloads/cs/DATA/split_data/test"
    
    # hyperparameters - tuned these for best results
    MODEL_NAME = "efficientnet_b0"  # works great for medical images
    BATCH_SIZE = 16  # smaller is better for generalization
    NUM_EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4  # regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    GRAD_CLIP = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # heavy augmentation - really helps prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # skin images can be any orientation
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # random patches removed
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Classes ({num_classes}): {class_names}")
    
    # handle class imbalance with weighted sampling - rama's idea
    class_counts = Counter([label for _, label in train_dataset.samples])
    
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    # Model
    model = get_model(MODEL_NAME, num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing LR Scheduler - Better than StepLR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training tracking
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    def train_one_epoch(model, loader, criterion, optimizer, use_mixup=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply Mixup
            if use_mixup and random.random() > 0.5:  # 50% chance
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                # For accuracy calculation
                _, preds = torch.max(outputs, 1)
                correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
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
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)
    
    # Training loop
    print(f"\nTraining {MODEL_NAME} with improved techniques...")
    print("=" * 100)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_weights = model.state_dict()
            print(f"✓ Validation accuracy improved to {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without improvement.")
                break
    
    # Save best model
    save_path = os.path.join(script_dir, f"improved_{MODEL_NAME}.pth")
    torch.save(best_model_weights, save_path)
    print(f"\nBest model saved to {save_path}")
    
    # Load best model for evaluation
    model.load_state_dict(best_model_weights)
    
    # Final evaluation
    _, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"Best Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open(f"improved_{MODEL_NAME}_report.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{MODEL_NAME} - Confusion Matrix (Test Acc: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig(f"improved_{MODEL_NAME}_cm.png", dpi=150)
    plt.close()
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"improved_{MODEL_NAME}_curves.png", dpi=150)
    plt.close()
    
    print(f"\nAll results saved with prefix 'improved_{MODEL_NAME}_'")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
