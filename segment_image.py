import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def device_initialization():
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
        print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
              "give numerically different outputs and sometimes degraded performance on MPS.")
    return device

def view_image(img_path, view=False):
    image = Image.open(img_path).convert("RGB")
    if view:
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.show()
    return np.array(image)

def initialize_sam2(device):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = SAM2ImagePredictor(sam2_model)
    return model

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.random.rand(3).tolist() + [0.6]
    else:
        color = [30/255, 144/255, 255/255, 0.6] 
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    mask_image = mask[:, :, None] * np.array(color)[None, None, :]
    ax.imshow(mask_image, interpolation='none', alpha=0.6)

def process_image(image, model, random_color=False, input_boxes=None,input_points= None,input_label = None):
    try:
        model.set_image(image)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')
        
        if input_boxes is not None:
            input_boxes = np.asarray(input_boxes, dtype=np.float32)
        if input_points is not None:
            input_points = np.asarray(input_points, dtype=np.float32)
        if input_label is not None:
            input_label = np.asarray(input_label, dtype=np.int32)
        masks, scores, _ = model.predict(
            point_coords=input_points,
            point_labels=input_label,
            box=input_boxes,
            multimask_output=False,
        )
        if input_boxes is not None:
            for mask in masks:
                show_mask(mask.squeeze(0), ax, random_color)
        else:
            for mask in masks:
                show_mask(mask, ax, random_color)
        plt.show()
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    device = device_initialization()
    model = initialize_sam2(device)
    
    image2 = view_image('./assets/football.jpg', view=False)
    boxes = np.array([
        [434, 244, 512, 349],
        [357, 103, 1233, 771]
    ])
    process_image(image2, model, random_color=True, input_boxes=boxes)
    
    image1 = view_image('./assets/dozer.webp', view=False)
    points = np.array([[250, 200], [450, 200]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int32)
    process_image(image1, model, random_color=False, input_boxes=None,input_points=points,input_label=labels) 

if __name__ == '__main__':
    main()
