import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from io import BytesIO
from collections import Counter

# the 4 classes we're detecting
CLASS_NAMES = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

# where we keep all our trained model weights
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_FILES = {
    "densenet121": "best_densenet121.pth",
    "efficientnet_b0": "best_efficientnet_b0.pth",
    "mobilenet_v3_large": "best_mobilenetv3_large.pth",
    "resnet50": "best_resnet50.pth",
    "shufflenetv2": "best_shufflenetv2.pth",
    "squeezenet1_1": "squeezenet1_1_best.pth",
    "ghostnet_100": "ghostnet_100_best.pth",
}


class GradCAM:
    # generates those heatmap overlays showing what the model is looking at
    # pretty cool visualization technique

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # hook into the layer to grab activations and gradients
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

        # backprop through just the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # weight the activations by their gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive contributions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        # normalize to 0-1 range
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output


def get_target_layer(model, model_name):
    # each model has a different layer we want to visualize
    # these are the last conv layers before classification - rama
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


def load_model_and_modify(model_name: str, num_classes: int, device: torch.device):
    # loads pretrained architecture and swaps out the classifier head for our 4 classes
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
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
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


def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    # standard imagenet preprocessing - all our models expect this
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet stats
    ])

    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    original_image = np.array(image.resize((224, 224)))
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, original_image


def predict_single_model(model, input_tensor: torch.Tensor, device: torch.device):
    # run inference on a single model and return class + probabilities
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = output.argmax(dim=1).item()
    return pred, probs


def visualize_gradcam(original_img, cam, title="Grad-CAM"):
    # creates the overlay visualization with red/yellow highlighting important regions
    img_normalized = original_img.astype(np.float32) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0

    # blend heatmap with original - nst tweaked these weights
    overlay = heatmap * 0.4 + img_normalized * 0.6
    overlay = np.clip(overlay, 0, 1)

    return heatmap, overlay


def predict_ensemble(image_bytes: bytes, include_gradcam: bool = False) -> Dict:
    # runs prediction through all available models and ensembles the results - rama
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)

    # prep the image
    input_tensor, original_image = preprocess_image(image_bytes)

    # we'll collect predictions from each model
    all_preds = []
    all_probs = []
    all_gradcams = []
    model_predictions = {}

    for model_name, model_file in MODEL_FILES.items():
        model_path = MODEL_DIR / model_file

        if not model_path.exists():
            continue  # skip if model file not found

        try:
            # load up the model
            model = load_model_and_modify(model_name, num_classes, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # get this model's prediction
            pred, probs = predict_single_model(model, input_tensor, device)
            all_preds.append(pred)
            all_probs.append(probs)

            model_predictions[model_name] = {
                "prediction": CLASS_NAMES[pred],
                "confidence": float(probs[pred])
            }

            # generate visualization if they want it
            if include_gradcam:
                try:
                    target_layer = get_target_layer(model, model_name)
                    gradcam = GradCAM(model, target_layer)
                    cam, _ = gradcam.generate_cam(input_tensor.to(device), target_class=pred)
                    all_gradcams.append(cam)
                except Exception as e:
                    pass  # gradcam failed, not a big deal

        except Exception as e:
            continue  # model failed to load, move on

    if len(all_probs) == 0:
        raise RuntimeError("No models could be loaded for prediction")

    # average all the probability distributions - works better than voting - nst
    avg_probs = np.mean(all_probs, axis=0)
    ensemble_pred = np.argmax(avg_probs)
    ensemble_confidence = float(avg_probs[ensemble_pred])

    # also do majority voting as backup
    majority_pred = Counter(all_preds).most_common(1)[0][0]

    # sort by confidence
    top_indices = np.argsort(avg_probs)[::-1]
    top_predictions = [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(avg_probs[idx])
        }
        for idx in top_indices
    ]

    result = {
        "predicted_class": CLASS_NAMES[ensemble_pred],
        "confidence": ensemble_confidence,
        "top_predictions": top_predictions,
        "ensemble_method": "Average Probability",
        "majority_voting_class": CLASS_NAMES[majority_pred],
        "num_models": len(all_probs),
        "model_predictions": model_predictions,
        "model_version": "Multi-Model Ensemble v1.0",
    }

    # attach gradcam if we generated any
    if include_gradcam and len(all_gradcams) > 0:
        # average the heatmaps from all models
        avg_gradcam = np.mean(all_gradcams, axis=0)
        heatmap, overlay = visualize_gradcam(original_image, avg_gradcam)

        # encode as base64 so we can send it in json
        import base64
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')

        result["visualization"] = {
            "type": "gradcam_overlay",
            "image": overlay_base64,
            "num_models_visualized": len(all_gradcams)
        }

    return result


def predict(image_bytes: bytes) -> Dict:
    return predict_ensemble(image_bytes, include_gradcam=False)
