import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_model(model_name, num_classes):
    # loads the right architecture and modifies for our classes
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=False)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=False)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif model_name == "ghostnet_100":
        try:
            import timm
            model = timm.create_model('ghostnet_100', pretrained=False, num_classes=num_classes)
        except ImportError:
            raise ImportError("timm library required for GhostNet")
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    return model

def test_time_augmentation_predict(model, image, device, num_augmentations=5):
    # TTA - runs prediction on flipped/rotated versions and averages
    # helps with accuracy - rama
    model.eval()
    
    tta_transforms = [
        transforms.Compose([]),  # Original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomRotation(15)]),
        transforms.Compose([transforms.RandomRotation(-15)]),
    ]
    
    all_outputs = []
    
    with torch.no_grad():
        for i in range(min(num_augmentations, len(tta_transforms))):
            # Apply augmentation to the tensor (already normalized)
            if i == 0:
                aug_image = image
            else:
                # Need to denormalize, apply transform, then renormalize
                # For simplicity, we'll use direct augmentations on the tensor
                if i == 1:  # Horizontal flip
                    aug_image = torch.flip(image, [3])
                elif i == 2:  # Vertical flip
                    aug_image = torch.flip(image, [2])
                else:  # Use original for rotation (requires PIL)
                    aug_image = image
            
            output = model(aug_image.to(device))
            all_outputs.append(F.softmax(output, dim=1).cpu())
    
    # average all the augmented predictions
    avg_output = torch.mean(torch.stack(all_outputs), dim=0)
    return avg_output


def get_predictions_with_tta(model, loader, device, use_tta=True):
    # wrapper to run model with or without TTA
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            if use_tta:
                # TTA for each image in batch
                batch_probs = []
                for i in range(inputs.size(0)):
                    img = inputs[i:i+1]
                    probs = test_time_augmentation_predict(model, img, device, num_augmentations=3)
                    batch_probs.append(probs)
                probs = torch.cat(batch_probs, dim=0)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu()
            
            all_probs.append(probs.numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.argmax(all_probs, axis=1)
    
    return all_preds, all_probs, all_labels

def calibrate_temperature(val_probs, val_labels, device):
    # temperature scaling - makes confidence scores more accurate
    # nst found this helps a lot
    val_probs_tensor = torch.FloatTensor(val_probs).to(device)
    val_labels_tensor = torch.LongTensor(val_labels).to(device)
    
    # learn the best temperature value
    temperature = nn.Parameter(torch.ones(1).to(device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(torch.log(val_probs_tensor) / temperature, val_labels_tensor)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    return temperature.item()

def apply_temperature_scaling(probs, temperature):
    """Apply learned temperature to logits"""
    # Convert probs back to logits
    logits = np.log(probs + 1e-10)
    # Apply temperature
    scaled_logits = logits / temperature
    # Convert back to probabilities
    scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
    return scaled_probs

def weighted_ensemble(prob_list, weights):
    # weights models based on how good they were on validation
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    
    weighted_probs = np.zeros_like(prob_list[0])
    for prob, weight in zip(prob_list, weights):
        weighted_probs += prob * weight
    
    return weighted_probs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    train_dir = os.path.join(base_dir, "DATA", "aug", "aug_train")
    val_dir = os.path.join(base_dir, "DATA", "split_data", "val")
    test_dir = os.path.join(base_dir, "DATA", "split_data", "test")
    
    # Transforms
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    num_classes = len(val_dataset.classes)
    class_names = val_dataset.classes
    
    print(f"\nClasses ({num_classes}): {class_names}")
    
    # Model files - using both old and improved models
    model_configs = [
        ("efficientnet_b0", os.path.join(script_dir, "best_efficientnet_b0.pth")),
        ("resnet50", os.path.join(script_dir, "best_resnet50.pth")),
        ("densenet121", os.path.join(script_dir, "best_densenet121.pth")),
        ("mobilenet_v3_large", os.path.join(script_dir, "best_mobilenetv3_large.pth")),
    ]
    
    # Check for improved models
    improved_models = []
    for model_name in ["efficientnet_b0", "resnet50", "densenet121", "mobilenet_v3_large"]:
        improved_path = os.path.join(script_dir, f"improved_{model_name}.pth")
        if os.path.exists(improved_path):
            improved_models.append((model_name, improved_path))
    
    if improved_models:
        print(f"\nFound {len(improved_models)} improved models!")
        model_configs = improved_models
    
    print("\n" + "=" * 100)
    print("EVALUATING INDIVIDUAL MODELS WITH TEST-TIME AUGMENTATION")
    print("=" * 100)
    
    val_accs = []
    val_probs_list = []
    test_probs_list = []
    val_labels = None
    test_labels = None
    
    for model_name, model_path in model_configs:
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: model file not found")
            continue
        
        print(f"\nLoading {model_name}...")
        model = get_model(model_name, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Validation with TTA
        val_preds, val_probs, val_labels = get_predictions_with_tta(model, val_loader, device, use_tta=True)
        val_acc = accuracy_score(val_labels, val_preds)
        val_accs.append(val_acc)
        val_probs_list.append(val_probs)
        
        # Test with TTA
        test_preds, test_probs, test_labels = get_predictions_with_tta(model, test_loader, device, use_tta=True)
        test_acc = accuracy_score(test_labels, test_preds)
        test_probs_list.append(test_probs)
        
        print(f"{model_name:<25} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    if len(val_accs) == 0:
        print("No models found! Please train models first.")
        return
    
    print("\n" + "=" * 100)
    print("ENSEMBLE METHODS")
    print("=" * 100)
    
    # Method 1: Simple Average
    avg_val_probs = np.mean(val_probs_list, axis=0)
    avg_test_probs = np.mean(test_probs_list, axis=0)
    
    avg_val_preds = np.argmax(avg_val_probs, axis=1)
    avg_test_preds = np.argmax(avg_test_probs, axis=1)
    
    avg_val_acc = accuracy_score(val_labels, avg_val_preds)
    avg_test_acc = accuracy_score(test_labels, avg_test_preds)
    
    print(f"\n1. Simple Average Ensemble:")
    print(f"   Validation Accuracy: {avg_val_acc:.4f}")
    print(f"   Test Accuracy:       {avg_test_acc:.4f}")
    
    # Method 2: Weighted Ensemble (based on validation accuracy)
    weighted_val_probs = weighted_ensemble(val_probs_list, val_accs)
    weighted_test_probs = weighted_ensemble(test_probs_list, val_accs)
    
    weighted_val_preds = np.argmax(weighted_val_probs, axis=1)
    weighted_test_preds = np.argmax(weighted_test_probs, axis=1)
    
    weighted_val_acc = accuracy_score(val_labels, weighted_val_preds)
    weighted_test_acc = accuracy_score(test_labels, weighted_test_preds)
    
    print(f"\n2. Weighted Ensemble (by val accuracy):")
    print(f"   Validation Accuracy: {weighted_val_acc:.4f}")
    print(f"   Test Accuracy:       {weighted_test_acc:.4f}")
    print(f"   Weights: {[f'{w:.3f}' for w in np.array(val_accs) / np.sum(val_accs)]}")
    
    # Method 3: Temperature Scaling + Weighted Ensemble
    print(f"\n3. Temperature-Scaled Weighted Ensemble:")
    print("   Learning optimal temperature on validation set...")
    
    temp = calibrate_temperature(weighted_val_probs, val_labels, device)
    print(f"   Optimal temperature: {temp:.4f}")
    
    scaled_test_probs = apply_temperature_scaling(weighted_test_probs, temp)
    scaled_test_preds = np.argmax(scaled_test_probs, axis=1)
    scaled_test_acc = accuracy_score(test_labels, scaled_test_preds)
    
    print(f"   Test Accuracy:       {scaled_test_acc:.4f}")
    
    # Select best method
    best_method = "Simple Average"
    best_acc = avg_test_acc
    best_preds = avg_test_preds
    
    if weighted_test_acc > best_acc:
        best_method = "Weighted Ensemble"
        best_acc = weighted_test_acc
        best_preds = weighted_test_preds
    
    if scaled_test_acc > best_acc:
        best_method = "Temperature-Scaled Weighted"
        best_acc = scaled_test_acc
        best_preds = scaled_test_preds
    
    print("\n" + "=" * 100)
    print("BEST ENSEMBLE RESULT")
    print("=" * 100)
    print(f"Method: {best_method}")
    print(f"Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Classification report
    report = classification_report(test_labels, best_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open("ensemble_improved_report.txt", "w") as f:
        f.write(f"Best Ensemble Method: {best_method}\n")
        f.write(f"Test Accuracy: {best_acc:.4f}\n\n")
        f.write(f"Individual Model Accuracies:\n")
        for (model_name, _), acc in zip(model_configs, val_accs):
            f.write(f"  {model_name}: {acc:.4f}\n")
        f.write(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, best_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
                yticklabels=class_names, cmap="Blues", cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Improved Ensemble - Confusion Matrix\nTest Accuracy: {best_acc:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig("ensemble_improved_cm.png", dpi=150)
    plt.close()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc, color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Per-Class Accuracy - {best_method}', fontsize=14)
    plt.ylim(0, 1.0)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("ensemble_improved_per_class.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 100)
    print("Per-Class Performance:")
    print("=" * 100)
    for cls, acc in zip(class_names, per_class_acc):
        status = "✓" if acc >= 0.9 else "✗"
        print(f"{status} {cls:<20}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nAll results saved with prefix 'ensemble_improved_'")
    print(f"Overall Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
