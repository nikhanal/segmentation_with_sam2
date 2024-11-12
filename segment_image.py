import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

points, labels = [], []

def device_initialization():
    """Initializes and returns the appropriate device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\nWarning: MPS support is preliminary. SAM2 may produce different outputs and reduced performance on MPS.")
    return device

def initialize_sam2(device):
    """Loads and initializes the SAM2 model on the given device."""
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)

def mouse_callback(event, x, y, flags, param):
    """Callback function to record clicked points with labels."""
    if event == cv2.EVENT_LBUTTONDOWN: 
        label = int(input(f"Enter label for point ({x}, {y}): "))
        points.append((x, y))
        labels.append(label)
        print(f"Point ({x}, {y}) with label {label} recorded.")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1) 
        cv2.imshow("Image", image)

def capture_points(image_path, output_file="points_labels.txt"):
    global image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at '{image_path}'")
        return

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open(output_file, "w") as f:
        for point, label in zip(points, labels):
            f.write(f"{point[0]},{point[1]},{label}\n")
    print(f"Points and labels saved to {output_file}")

def load_points_labels(file_path="points_labels.txt"):
    """Loads points and labels from a text file."""
    points, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            x, y, label = line.strip().split(',')
            points.append([float(x), float(y)])
            labels.append(int(label))
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

def view_image(img_path, display=False):
    """Loads an image and optionally displays it using Matplotlib."""
    image = Image.open(img_path).convert("RGB")
    if display:
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.show()
    return np.array(image)

def show_mask(mask, ax, random_color=True):
    """Displays a mask overlay on the given Matplotlib axis."""
    color = np.random.rand(3).tolist() + [0.6] if random_color else [30/255, 144/255, 255/255, 0.6]
    mask_image = mask.astype(np.uint8)[:, :, None] * np.array(color)[None, None, :]
    ax.imshow(mask_image, interpolation='none', alpha=0.6)

def process_image(image, model, random_color=False, input_boxes=None, input_points=None, input_labels=None):
    """Processes an image using SAM2 model and displays the segmentation results."""
    try:
        model.set_image(image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        input_points = np.asarray(input_points, dtype=np.float32) if input_points is not None else None
        input_labels = np.asarray(input_labels, dtype=np.int32) if input_labels is not None else None
        input_boxes = np.asarray(input_boxes, dtype=np.float32) if input_boxes is not None else None

        masks, scores, _ = model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=False,
        )
        
        print(f"Shape of masks: {masks.shape}")

        for mask in masks:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            show_mask(mask, ax, random_color)
        
        plt.show()
    except Exception as e:
        print(f"Error processing image: {e}")

def main(image_paths):
    device = device_initialization()
    model = initialize_sam2(device)

    for img_path in image_paths:
        output_file = os.path.splitext(img_path)[0] + "_points_labels.txt"
        capture_points(img_path, output_file=output_file)
        
        points, labels = load_points_labels(output_file)
        
        image = view_image(img_path, display=False)
        process_image(image, model, random_color=False, input_boxes=None, input_points=points, input_labels=labels)

if __name__ == '__main__':
    image_paths = ['../files/dozer.webp']  
    main(image_paths)
