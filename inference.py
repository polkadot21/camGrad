import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for an image and a CNN model")
    parser.add_argument("--image_path", type=str, required=True, help="path to input image file")
    parser.add_argument("--model_path", type=str, default="resnet50.pth", help="path to CNN model file (default: pre-trained ResNet50)")
    return parser.parse_args()

def main():
    args = get_args()

    if args.model_path == "resnet50.pth":
        model = models.resnet50(pretrained=True)
    else:
        model = torch.load(args.model_path)

    model.eval()

    # Define the target layer for Grad-CAM
    target_layer = model.layer4[-1]

    transform = nn.Sequential(
        nn.Resize((224, 224)),
        nn.ToTensor(),
        nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    image = cv2.imread(args.image_path)

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    features = target_layer(image_tensor)

    class_score = features.sum(dim=[2,3])
    class_score.backward(torch.ones_like(class_score))

    weights = torch.mean(target_layer.weight.grad, dim=[2,3])

    # Multiply the feature maps by their corresponding weights and sum them up
    weighted_features = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * features, dim=[0,1])

    cam = nn.ReLU(inplace=True)(weighted_features)

    # Upsample the feature map to the size of the input image
    cam = nn.functional.interpolate(cam.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=False)
    cam = cam.squeeze(0)

    # Normalize the feature map
    cam = cam.detach().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(255 * cam)

    # Apply color mapping to the feature map
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Overlay the heatmap onto the input image
    alpha = 0.5
    output_image = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

    output_path = "grad_cam.png"
    cv2.imwrite(output_path, output_image)
    print("Grad-CAM heatmap saved to", output_path)

if __name__ == '__main__':
    main()