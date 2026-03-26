import os
import cv2
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

# --- 1. DATA CLASSES (From grounding_dino.py) ---

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )

# --- 2. THE ORIGINAL SAVE FUNCTIONS (Required to see output!) ---

def save_outline_pngs(image, detections, save_prefix, outline_thickness=3):
    """Saves transparent outline PNGs."""
    height, width = image.shape[:2]
    for idx, detection in enumerate(detections):
        if detection.mask is not None:
            contours, _ = cv2.findContours(detection.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                temp_image = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.drawContours(temp_image, contours, -1, (0, 0, 255), thickness=outline_thickness)
                outline_image = np.zeros((height, width, 4), dtype=np.uint8)
                outline_image[:,:,:3] = temp_image
                outline_mask = np.any(temp_image > 0, axis=2)
                outline_image[:,:,3] = outline_mask.astype(np.uint8) * 255
                cv2.imwrite(f"{save_prefix}_{idx:03d}.png", outline_image)

def save_cutout_pngs(image, detections, save_prefix):
    """Saves original pixels with transparent background."""
    height, width = image.shape[:2]
    for idx, detection in enumerate(detections):
        if detection.mask is not None:
            cutout_image = np.zeros((height, width, 4), dtype=np.uint8)
            mask_area = detection.mask > 0
            # Convert RGB to BGR for OpenCV saving
            cutout_image[mask_area, :3] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[mask_area]
            cutout_image[mask_area, 3] = 255
            cv2.imwrite(f"{save_prefix}_{idx:03d}.png", cutout_image)

# --- 3. OPTIMIZED PIPELINE FUNCTIONS ---

def detect_batch(image, labels, detector_pipe, threshold=0.4):
    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = detector_pipe(image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(result) for result in results]

def segment_batch(image, detections, segmentator, processor, device):
    if not detections: return []
    boxes = [[d.box.xyxy for d in detections]]
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segmentator(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes)[0]
    masks_np = (masks.cpu().float().permute(0, 2, 3, 1).mean(axis=-1) > 0).int().numpy().astype(np.uint8)
    for idx, mask in enumerate(list(masks_np)):
        detections[idx].mask = mask
    return detections

# --- 4. THE EXECUTOR ---

def async_save_worker(image_array, detections, output_dir, frame_name):
    """Wraps the individual save functions."""
    prefix = os.path.join(output_dir, frame_name)
    # save_outline_pngs(image_array, detections, f"{prefix}_outline")
    save_cutout_pngs(image_array, detections, f"{prefix}_cutout")

def main_loop(input_dir, output_dir, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Models onto {device}...")

    det_pipe = pipeline(model="IDEA-Research/grounding-dino-base", task="zero-shot-object-detection", device=device)
    seg_model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-base").to(device)
    seg_proc = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for filename in tqdm(files, desc="Batch Processing"):
            img = Image.open(os.path.join(input_dir, filename)).convert("RGB")

            # Detect & Segment
            results = detect_batch(img, labels, det_pipe)
            if results:
                results = segment_batch(img, results, seg_model, seg_proc, device)

                # Async Save
                img_arr = np.array(img)
                name = os.path.splitext(filename)[0]
                executor.submit(async_save_worker, img_arr, results, output_dir, name)

if __name__ == "__main__":
    INPUT = "../build"
    OUTPUT = "processed_output"
    TARGETS = ["human"]

    main_loop(INPUT, OUTPUT, TARGETS)
