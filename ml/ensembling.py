import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=1):
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_loader, test_loader, val_dataset.classes

def load_model_and_modify(model_name, num_classes, device):
    if model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=False)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

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
            raise ImportError("timm library is required for GhostNet. Install with: pip install timm")

    else:
        raise ValueError(f"Model name {model_name} not recognized")

    model = model.to(device)
    return model

def calculate_auc(labels, probs, num_classes):
                                                 
    try:
                                         
        labels_bin = label_binarize(labels, classes=range(num_classes))
        
                                             
        auc_macro = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
        
                                                 
        auc_weighted = roc_auc_score(labels_bin, probs, average='weighted', multi_class='ovr')
        
        return auc_macro, auc_weighted
    except Exception as e:
        print(f"Warning: Could not calculate AUC - {e}")
        return None, None

def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = correct / total
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auc_macro, auc_weighted = calculate_auc(all_labels, all_probs, num_classes)
    
    return acc, np.array(all_preds), all_labels, all_probs, auc_macro, auc_weighted

                  
def majority_voting(predictions_list):
    predictions_stack = np.stack(predictions_list, axis=1)
    final_preds = []
    for preds_per_sample in predictions_stack:
        vote_counts = Counter(preds_per_sample)
        final_preds.append(vote_counts.most_common(1)[0][0])
    return np.array(final_preds)

def average_probability(prob_list):
    avg_prob = np.mean(prob_list, axis=0)
    return np.argmax(avg_prob, axis=1), avg_prob

def weighted_average_probability(prob_list, weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted_prob = np.zeros_like(prob_list[0])
    for prob, w in zip(prob_list, weights):
        weighted_prob += prob * w
    return np.argmax(weighted_prob, axis=1), weighted_prob

def weighted_voting(predictions_list, weights):
    predictions_stack = np.stack(predictions_list, axis=1)
    weights = np.array(weights)
    weights = weights / weights.sum()

    final_preds = []
    for preds_per_sample in predictions_stack:
        vote_score = {}
        for pred, w in zip(preds_per_sample, weights):
            vote_score[pred] = vote_score.get(pred, 0) + w
        final_preds.append(max(vote_score, key=vote_score.get))
    return np.array(final_preds)

def max_probability_voting(prob_list, predictions_list):
    prob_stack = np.stack(prob_list, axis=1)
    preds_stack = np.stack(predictions_list, axis=1)

    final_preds = []
    final_probs = []
    for sample_probs, sample_preds in zip(prob_stack, preds_stack):
        max_probs_per_model = sample_probs.max(axis=1)
        max_model_idx = np.argmax(max_probs_per_model)
        final_preds.append(sample_preds[max_model_idx])
        final_probs.append(sample_probs[max_model_idx])
    return np.array(final_preds), np.array(final_probs)

                               
def shannon_entropy(probs):
    epsilon = 1e-10
    return -np.sum(probs * np.log(probs + epsilon), axis=-1)



def gini_index(probs):
    return 1 - np.sum(probs**2, axis=-1)

                                        
def entropy_based_majority_voting_remove_models(predictions_list, prob_list, entropy_func, num_remove=1, **entropy_kwargs):
    predictions_stack = np.stack(predictions_list, axis=1)
    prob_stack = np.stack(prob_list, axis=1)
    num_models = predictions_stack.shape[1]
    
    if num_remove >= num_models:
        raise ValueError(f"Cannot remove {num_remove} models from {num_models} total models")

    final_preds = []
    for i in range(predictions_stack.shape[0]):
        sample_probs = prob_stack[i]
        sample_preds = predictions_stack[i]
        sample_entropy = entropy_func(sample_probs, **entropy_kwargs)
        
        if num_remove == 1:
            exclude_idx = np.argmax(sample_entropy)
            include_idxs = [idx for idx in range(num_models) if idx != exclude_idx]
        else:
            exclude_idxs = np.argsort(sample_entropy)[-num_remove:]
            include_idxs = [idx for idx in range(num_models) if idx not in exclude_idxs]

        votes = sample_preds[include_idxs]
        vote_counts = Counter(votes)
        final_preds.append(vote_counts.most_common(1)[0][0])
    return np.array(final_preds)

def entropy_based_average_probability_remove_models(prob_list, entropy_func, num_remove=1, **entropy_kwargs):
    prob_stack = np.stack(prob_list, axis=1)
    num_samples, num_models, num_classes = prob_stack.shape
    
    if num_remove >= num_models:
        raise ValueError(f"Cannot remove {num_remove} models from {num_models} total models")

    final_preds = []
    final_probs = []
    for i in range(num_samples):
        sample_probs = prob_stack[i]
        sample_entropy = entropy_func(sample_probs, **entropy_kwargs)
        
        if num_remove == 1:
            exclude_idx = np.argmax(sample_entropy)
            include_idxs = [idx for idx in range(num_models) if idx != exclude_idx]
        else:
            exclude_idxs = np.argsort(sample_entropy)[-num_remove:]
            include_idxs = [idx for idx in range(num_models) if idx not in exclude_idxs]
            
        avg_prob = np.mean(sample_probs[include_idxs], axis=0)
        final_preds.append(np.argmax(avg_prob))
        final_probs.append(avg_prob)
    return np.array(final_preds), np.array(final_probs)

def entropy_based_weighted_voting(predictions_list, prob_list, entropy_func, **entropy_kwargs):
    predictions_stack = np.stack(predictions_list, axis=1)
    prob_stack = np.stack(prob_list, axis=1)
    num_samples, num_models = predictions_stack.shape
    final_preds = []

    for i in range(num_samples):
        sample_probs = prob_stack[i]
        sample_preds = predictions_stack[i]
        sample_entropy = entropy_func(sample_probs, **entropy_kwargs)
        
        weights = 1 / (sample_entropy + 1e-10)
        weights = weights / weights.sum()

        vote_score = {}
        for pred, w in zip(sample_preds, weights):
            vote_score[pred] = vote_score.get(pred, 0) + w
        final_preds.append(max(vote_score, key=vote_score.get))
    return np.array(final_preds)

                                   
def val_acc_based_majority_voting_remove_one(predictions_list, val_accs, model_names):
    worst_model_idx = np.argmin(val_accs)
    filtered_predictions = [pred for i, pred in enumerate(predictions_list) if i != worst_model_idx]
    
    print(f"  [Val Acc Based] Removing worst model: {model_names[worst_model_idx]} (Val Acc: {val_accs[worst_model_idx]:.4f})")
    
    return majority_voting(filtered_predictions)

def val_acc_based_average_probability_remove_one(prob_list, val_accs, model_names):
    worst_model_idx = np.argmin(val_accs)
    filtered_probs = [prob for i, prob in enumerate(prob_list) if i != worst_model_idx]
    
    print(f"  [Val Acc Based] Removing worst model: {model_names[worst_model_idx]} (Val Acc: {val_accs[worst_model_idx]:.4f})")
    
    return average_probability(filtered_probs)

def evaluate_ensemble_methods(predictions_list, prob_list, labels, num_classes, val_accs=None, model_names=None, method_prefix=""):
                                                                              
    results = {}
    
                      
    maj_preds = majority_voting(predictions_list)
    results[f"{method_prefix}Majority Voting"] = {
        'acc': (maj_preds == labels).mean(),
        'probs': None                                                 
    }
    
    avg_preds, avg_probs = average_probability(prob_list)
    auc_macro, auc_weighted = calculate_auc(labels, avg_probs, num_classes)
    results[f"{method_prefix}Average Probability"] = {
        'acc': (avg_preds == labels).mean(),
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'probs': avg_probs
    }
    
    if val_accs is not None:
        w_avg_preds, w_avg_probs = weighted_average_probability(prob_list, val_accs)
        auc_macro, auc_weighted = calculate_auc(labels, w_avg_probs, num_classes)
        results[f"{method_prefix}Weighted Average Probability"] = {
            'acc': (w_avg_preds == labels).mean(),
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted,
            'probs': w_avg_probs
        }
        
        w_vote_preds = weighted_voting(predictions_list, val_accs)
        results[f"{method_prefix}Weighted Voting"] = {
            'acc': (w_vote_preds == labels).mean(),
            'probs': None
        }
    
    max_prob_preds, max_prob_probs = max_probability_voting(prob_list, predictions_list)
    auc_macro, auc_weighted = calculate_auc(labels, max_prob_probs, num_classes)
    results[f"{method_prefix}Max Probability Voting"] = {
        'acc': (max_prob_preds == labels).mean(),
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'probs': max_prob_probs
    }
    
                                   
    for num_remove in [1, 2]:
        if len(predictions_list) > num_remove:
            shannon_maj_preds = entropy_based_majority_voting_remove_models(predictions_list, prob_list, shannon_entropy, num_remove)
            results[f"{method_prefix}Shannon Majority (remove {num_remove})"] = {
                'acc': (shannon_maj_preds == labels).mean(),
                'probs': None
            }
            
            shannon_avg_preds, shannon_avg_probs = entropy_based_average_probability_remove_models(prob_list, shannon_entropy, num_remove)
            auc_macro, auc_weighted = calculate_auc(labels, shannon_avg_probs, num_classes)
            results[f"{method_prefix}Shannon Avg Prob (remove {num_remove})"] = {
                'acc': (shannon_avg_preds == labels).mean(),
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted,
                'probs': shannon_avg_probs
            }
    
    shannon_wv_preds = entropy_based_weighted_voting(predictions_list, prob_list, shannon_entropy)
    results[f"{method_prefix}Shannon Weighted Voting"] = {
        'acc': (shannon_wv_preds == labels).mean(),
        'probs': None
    }
    
                              
    for num_remove in [1, 2]:
        if len(predictions_list) > num_remove:
            g_maj_preds = entropy_based_majority_voting_remove_models(predictions_list, prob_list, gini_index, num_remove)
            results[f"{method_prefix}Gini Majority (remove {num_remove})"] = {
                'acc': (g_maj_preds == labels).mean(),
                'probs': None
            }
            
            g_avg_preds, g_avg_probs = entropy_based_average_probability_remove_models(prob_list, gini_index, num_remove)
            auc_macro, auc_weighted = calculate_auc(labels, g_avg_probs, num_classes)
            results[f"{method_prefix}Gini Avg Prob (remove {num_remove})"] = {
                'acc': (g_avg_preds == labels).mean(),
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted,
                'probs': g_avg_probs
            }
    
    g_wv_preds = entropy_based_weighted_voting(predictions_list, prob_list, gini_index)
    results[f"{method_prefix}Gini Weighted Voting"] = {
        'acc': (g_wv_preds == labels).mean(),
        'probs': None
    }
    
                                       
    if val_accs is not None and model_names is not None:
        va_maj_preds = val_acc_based_majority_voting_remove_one(predictions_list, val_accs, model_names)
        results[f"{method_prefix}Val Acc Majority (remove 1)"] = {
            'acc': (va_maj_preds == labels).mean(),
            'probs': None
        }
        
        va_avg_preds, va_avg_probs = val_acc_based_average_probability_remove_one(prob_list, val_accs, model_names)
        auc_macro, auc_weighted = calculate_auc(labels, va_avg_probs, num_classes)
        results[f"{method_prefix}Val Acc Avg Prob (remove 1)"] = {
            'acc': (va_avg_preds == labels).mean(),
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted,
            'probs': va_avg_probs
        }
    
    return results

def print_results(results, title):
                                                        
    print(f"\n{title}")
    print("=" * 100)
    
                                  
    standard_methods = {k: v for k, v in results.items() if any(x in k for x in ['Majority Voting', 'Average Probability', 'Weighted', 'Max Probability']) and 'Shannon' not in k and 'Gini' not in k and 'Val Acc' not in k}
    shannon_methods = {k: v for k, v in results.items() if 'Shannon' in k}
    gini_methods = {k: v for k, v in results.items() if 'Gini' in k}
    val_acc_methods = {k: v for k, v in results.items() if 'Val Acc' in k}
    
    def print_group(methods, group_name):
        if methods:
            print(f"\n{group_name}:")
            for method, metrics in sorted(methods.items()):
                if isinstance(metrics, dict):
                    acc = metrics.get('acc', 0)
                    auc_macro = metrics.get('auc_macro')
                    auc_weighted = metrics.get('auc_weighted')
                    
                    if auc_macro is not None:
                        print(f"{method:<55}: Acc={acc:.4f} | AUC(Macro)={auc_macro:.4f} | AUC(Weighted)={auc_weighted:.4f}")
                    else:
                        print(f"{method:<55}: Acc={acc:.4f}")
                else:
                    print(f"{method:<55}: {metrics:.4f}")
    
    print_group(standard_methods, "STANDARD METHODS")
    print_group(shannon_methods, "SHANNON ENTROPY-BASED METHODS")
    print_group(gini_methods, "GINI INDEX-BASED METHODS")
    print_group(val_acc_methods, "VALIDATION ACCURACY-BASED METHODS")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent directory (cs)
    
    train_dir = os.path.join(base_dir, "DATA", "aug", "aug_train")
    val_dir = os.path.join(base_dir, "DATA", "split_data", "val")
    test_dir = os.path.join(base_dir, "DATA", "split_data", "test")

    val_loader, test_loader, class_names = get_dataloaders(train_dir, val_dir, test_dir)
    num_classes = len(class_names)

    model_files = {
        "densenet121": os.path.join(script_dir, "best_densenet121.pth"),
        "efficientnet_b0": os.path.join(script_dir, "best_efficientnet_b0.pth"),
        "mobilenet_v3_large": os.path.join(script_dir, "best_mobilenetv3_large.pth"),
        "resnet50": os.path.join(script_dir, "best_resnet50.pth"),
        "shufflenetv2": os.path.join(script_dir, "best_shufflenetv2.pth"),
        "squeezenet1_1": os.path.join(script_dir, "squeezenet1_1_best.pth"),
        "ghostnet_100" : os.path.join(script_dir, "ghostnet_100_best.pth"),
    }

    results = {}

                                         
    print("\n" + "="*100)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*100)
    
    for model_name, path in model_files.items():
        print(f"\nLoading model: {model_name} from {path}")
        model = load_model_and_modify(model_name, num_classes, device)
        model.load_state_dict(torch.load(path, map_location=device))

        val_acc, val_preds, val_labels, val_probs, val_auc_macro, val_auc_weighted = evaluate(model, val_loader, device, num_classes)
        test_acc, test_preds, test_labels, test_probs, test_auc_macro, test_auc_weighted = evaluate(model, test_loader, device, num_classes)

        print(f"{model_name} Validation - Acc: {val_acc:.4f} | AUC(Macro): {val_auc_macro:.4f} | AUC(Weighted): {val_auc_weighted:.4f}")
        print(f"{model_name} Test       - Acc: {test_acc:.4f} | AUC(Macro): {test_auc_macro:.4f} | AUC(Weighted): {test_auc_weighted:.4f}")

        results[model_name] = {
            "val_acc": val_acc,
            "val_auc_macro": val_auc_macro,
            "val_auc_weighted": val_auc_weighted,
            "val_preds": val_preds,
            "val_labels": val_labels,
            "val_probs": val_probs,
            "test_acc": test_acc,
            "test_auc_macro": test_auc_macro,
            "test_auc_weighted": test_auc_weighted,
            "test_preds": test_preds,
            "test_labels": test_labels,
            "test_probs": test_probs,
        }

                                       
    val_preds_list = [results[m]['val_preds'] for m in model_files]
    val_probs_list = [results[m]['val_probs'] for m in model_files]
    val_accs = [results[m]['val_acc'] for m in model_files]
    model_names = list(model_files.keys())

    test_preds_list = [results[m]['test_preds'] for m in model_files]
    test_probs_list = [results[m]['test_probs'] for m in model_files]

    val_labels = results[next(iter(model_files))]['val_labels']
    test_labels = results[next(iter(model_files))]['test_labels']

                                            
    print("\n" + "="*100)
    print("INDIVIDUAL MODEL SUMMARY")
    print("="*100)
    print(f"{'Model':<20} | {'Val Acc':<8} | {'Val AUC(M)':<11} | {'Val AUC(W)':<11} | {'Test Acc':<8} | {'Test AUC(M)':<11} | {'Test AUC(W)':<11}")
    print("-"*100)
    for model_name in model_files:
        print(f"{model_name:<20} | {results[model_name]['val_acc']:.4f}   | "
              f"{results[model_name]['val_auc_macro']:.4f}      | "
              f"{results[model_name]['val_auc_weighted']:.4f}      | "
              f"{results[model_name]['test_acc']:.4f}   | "
              f"{results[model_name]['test_auc_macro']:.4f}      | "
              f"{results[model_name]['test_auc_weighted']:.4f}")
    print("="*100)

                                                 
    val_results = evaluate_ensemble_methods(
        val_preds_list, val_probs_list, val_labels, num_classes, val_accs, model_names, "Val_"
    )
    
                                           
    test_results = evaluate_ensemble_methods(
        test_preds_list, test_probs_list, test_labels, num_classes, val_accs, model_names, "Test_"
    )

                   
    print_results({k.replace("Val_", ""): v for k, v in val_results.items()}, "ENSEMBLE VALIDATION RESULTS")
    print_results({k.replace("Test_", ""): v for k, v in test_results.items()}, "ENSEMBLE TEST RESULTS")
    
                                 
    print("\n" + "="*100)
    print("BEST PERFORMING METHODS")
    print("="*100)
    
                      
    val_best_acc = max(val_results.items(), key=lambda x: x[1]['acc'] if isinstance(x[1], dict) else x[1])
    test_best_acc = max(test_results.items(), key=lambda x: x[1]['acc'] if isinstance(x[1], dict) else x[1])
    
    print(f"\nBest Validation Accuracy:")
    print(f"  Method: {val_best_acc[0].replace('Val_', '')}")
    if isinstance(val_best_acc[1], dict):
        print(f"  Accuracy: {val_best_acc[1]['acc']:.4f}")
        if val_best_acc[1].get('auc_macro'):
            print(f"  AUC (Macro): {val_best_acc[1]['auc_macro']:.4f}")
            print(f"  AUC (Weighted): {val_best_acc[1]['auc_weighted']:.4f}")
    else:
        print(f"  Accuracy: {val_best_acc[1]:.4f}")
    
    print(f"\nBest Test Accuracy:")
    print(f"  Method: {test_best_acc[0].replace('Test_', '')}")
    if isinstance(test_best_acc[1], dict):
        print(f"  Accuracy: {test_best_acc[1]['acc']:.4f}")
        if test_best_acc[1].get('auc_macro'):
            print(f"  AUC (Macro): {test_best_acc[1]['auc_macro']:.4f}")
            print(f"  AUC (Weighted): {test_best_acc[1]['auc_weighted']:.4f}")
    else:
        print(f"  Accuracy: {test_best_acc[1]:.4f}")
    
                                             
    val_with_auc = {k: v for k, v in val_results.items() if isinstance(v, dict) and v.get('auc_macro') is not None}
    test_with_auc = {k: v for k, v in test_results.items() if isinstance(v, dict) and v.get('auc_macro') is not None}
    
    if val_with_auc:
        val_best_auc = max(val_with_auc.items(), key=lambda x: x[1]['auc_macro'])
        print(f"\nBest Validation AUC (Macro):")
        print(f"  Method: {val_best_auc[0].replace('Val_', '')}")
        print(f"  Accuracy: {val_best_auc[1]['acc']:.4f}")
        print(f"  AUC (Macro): {val_best_auc[1]['auc_macro']:.4f}")
        print(f"  AUC (Weighted): {val_best_auc[1]['auc_weighted']:.4f}")
    
    if test_with_auc:
        test_best_auc = max(test_with_auc.items(), key=lambda x: x[1]['auc_macro'])
        print(f"\nBest Test AUC (Macro):")
        print(f"  Method: {test_best_auc[0].replace('Test_', '')}")
        print(f"  Accuracy: {test_best_auc[1]['acc']:.4f}")
        print(f"  AUC (Macro): {test_best_auc[1]['auc_macro']:.4f}")
        print(f"  AUC (Weighted): {test_best_auc[1]['auc_weighted']:.4f}")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()