import json  # Import the json module

# ...
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from flask import jsonify


from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def pred_and_return_results(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model and returns the results.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    
    Returns:
        dict: A dictionary containing prediction results including label, probability, and advisory points.
    """
    
    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Prepare the result as a dictionary
    result = {
        "label": class_names[target_image_pred_label],
        "probability": target_image_pred_probs.max().item(),
        "advisory_points": []  # Initialize advisory points as an empty list
    }

    # Add advisory points based on the predicted class label
    if class_names[target_image_pred_label] == "Acne_Rosacea":
        result["advisory_points"] = [
            "Gentle Cleansing: Use a mild, non-abrasive cleanser to clean your face gently. Avoid harsh scrubbing, which can irritate both conditions. [Link]"          
                       
        ]
    elif class_names[target_image_pred_label] == "Actinic":
        result["advisory_points"] = [
            "Hydrate Your Skin: Keep your skin well-hydrated with a gentle, fragrance-free moisturizer. Apply it immediately after bathing to lock in moisture. [Link]"
        ]
    elif class_names[target_image_pred_label] == "Atopic_Dermatitis":
        result["advisory_points"] = [
            "Sun Protection: Protect your skin from the sun by wearing sunscreen with a high SPF, protective clothing, and wide-brimmed hats. Limit sun exposure, especially during peak hours. [Link]"
            
        ]

    return result