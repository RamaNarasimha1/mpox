import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None


        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)


        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)


        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output

def get_target_layer(model, model_name):

    if model_name == "densenet121":
        return model.features.denseblock4.denselayer16.conv2
    elif model_name == "efficientnet_b0":
        return model.features[-1][0]
    elif model_name == "mobilenet_v3_large":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1].conv3
    elif model_name == "shufflenetv2":
        return model.conv5
    elif model_name == "squeezenet1_1":
        return model.features[-1]
    elif model_name == "ghostnet_100":
        return model.blocks[-1][-1]
    else:
        raise ValueError(f"Model {model_name} not supported for Grad-CAM")

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
            raise ImportError("timm library required for GhostNet. Install: pip install timm")
    else:
        raise ValueError(f"Model {model_name} not recognized")

    model = model.to(device)
    return model

def preprocess_image(image_path):


    image_path = os.path.normpath(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize((224, 224)))
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, original_image

def predict_single_model(model, input_tensor, device):

    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = output.argmax(dim=1).item()
    return pred, probs

def visualize_gradcam(original_img, cam, title="Grad-CAM"):


    img_normalized = original_img.astype(np.float32) / 255.0


    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0


    overlay = heatmap * 0.4 + img_normalized * 0.6
    overlay = np.clip(overlay, 0, 1)

    return heatmap, overlay

def predict_and_visualize(image_path, model_files, class_names, device):

    print(f"\nProcessing image: {image_path}")
    print("=" * 100)


    input_tensor, original_image = preprocess_image(image_path)
    num_classes = len(class_names)


    all_preds = []
    all_probs = []
    all_gradcams = []


    print("\nIndividual Model Predictions:")
    print("-" * 100)

    for model_name, model_path in model_files.items():

        model = load_model_and_modify(model_name, num_classes, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()


        pred, probs = predict_single_model(model, input_tensor, device)
        all_preds.append(pred)
        all_probs.append(probs)

        print(f"{model_name:<20}: {class_names[pred]:<25} (Confidence: {probs[pred]:.4f})")


        try:
            target_layer = get_target_layer(model, model_name)
            gradcam = GradCAM(model, target_layer)
            cam, _ = gradcam.generate_cam(input_tensor.to(device), target_class=pred)
            all_gradcams.append(cam)
        except Exception as e:
            print(f"  Warning: Could not generate Grad-CAM for {model_name}: {e}")


    print("\n" + "=" * 100)
    print("Ensemble Prediction (Average Probability):")
    print("-" * 100)

    avg_probs = np.mean(all_probs, axis=0)
    ensemble_pred = np.argmax(avg_probs)
    ensemble_class = class_names[ensemble_pred]
    ensemble_confidence = avg_probs[ensemble_pred]

    print(f"Predicted Class: {ensemble_class}")
    print(f"Confidence: {ensemble_confidence:.4f}")
    print("\nTop 3 Predictions:")
    top3_idx = np.argsort(avg_probs)[-3:][::-1]
    for i, idx in enumerate(top3_idx):
        print(f"  {i+1}. {class_names[idx]:<25}: {avg_probs[idx]:.4f}")


    if len(all_gradcams) > 0:
        avg_gradcam = np.mean(all_gradcams, axis=0)
        heatmap, overlay = visualize_gradcam(original_image, avg_gradcam)


        print("\n" + "=" * 100)
        print("Generating Visualization...")
        print("=" * 100)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))


        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')


        im = axes[1].imshow(avg_gradcam, cmap='jet')
        axes[1].set_title('Average Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)


        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')


        axes[3].axis('off')
        info_text = f"ENSEMBLE PREDICTION\n(Average Probability)\n\n"
        info_text += f"Predicted Class:\n{ensemble_class}\n\n"
        info_text += f"Confidence: {ensemble_confidence:.4f}\n\n"
        info_text += f"Top 3 Classes:\n"
        for i, idx in enumerate(top3_idx):
            info_text += f"{i+1}. {class_names[idx]}\n   {avg_probs[idx]:.4f}\n"

        axes[3].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('ensemble_prediction_gradcam.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'ensemble_prediction_gradcam.png'")
        plt.show()
    else:
        print("\nWarning: No Grad-CAM visualizations available")

    print("\n" + "=" * 100)
    print("Prediction complete!")
    print("=" * 100)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent directory (cs)

    model_files = {
        "densenet121": os.path.join(script_dir, "best_densenet121.pth"),
        "efficientnet_b0": os.path.join(script_dir, "best_efficientnet_b0.pth"),
        "mobilenet_v3_large": os.path.join(script_dir, "best_mobilenetv3_large.pth"),
        "resnet50": os.path.join(script_dir, "best_resnet50.pth"),
        "shufflenetv2": os.path.join(script_dir, "best_shufflenetv2.pth"),
        "squeezenet1_1": os.path.join(script_dir, "squeezenet1_1_best.pth"),
        "ghostnet_100": os.path.join(script_dir, "ghostnet_100_best.pth"),
    }

    val_dir = os.path.join(base_dir, "DATA", "split_data", "val")
    if os.path.exists(val_dir):
        dataset = datasets.ImageFolder(val_dir)
        class_names = dataset.classes
    else:

        class_names = ['class_0', 'class_1', 'class_2']

    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")


    image_path = input("\nEnter image path: ").strip().strip('"').strip("'")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return


    predict_and_visualize(image_path, model_files, class_names, device)

if __name__ == "__main__":
    main()
