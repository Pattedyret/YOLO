import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from omegaconf import OmegaConf
import argparse
import time
from typing import Tuple, List
from PIL import Image
import hydra
from hydra.core.hydra_config import HydraConfig

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.model.yolo import create_model
from yolo.config.config import Config
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.bounding_box_utils import create_converter
from yolo.tools.drawer import draw_bboxes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    return parser.parse_args()

def create_model_and_config(cfg: Config, args):
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device
    
    print(f"Loading weights from: {cfg.weight}")
    print(f"Using model configuration: {cfg.model.name}")
    
    # Create model and load weights
    model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
    model = model.to(cfg.device)
    if args.half:
        model = model.half()
    model.eval()
    
    return model, cfg, cfg.dataset.class_list

def convert_predictions_to_boxes(preds_cls, preds_box, conf_thres=0.25):
    # Get confidence scores and class indices
    conf, cls = torch.max(preds_cls, dim=-1)
    mask = conf > conf_thres
    
    # Filter predictions
    boxes = preds_box[mask]
    conf = conf[mask]
    cls = cls[mask]
    
    # Convert to list format [class_id, x_min, y_min, x_max, y_max, conf]
    boxes_list = []
    for box, c, conf_score in zip(boxes, cls, conf):
        boxes_list.append([c.item(), *box.tolist(), conf_score.item()])
    
    return boxes_list

def process_frame(frame: np.ndarray, model, converter, cfg, args) -> Tuple[np.ndarray, List[float]]:
    # Preprocess image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tuple(cfg.image_size))
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if args.half:
        img = img.half()
    img = img.to(cfg.device)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(img)
        predictions = predictions["Main"]  # Use only the main predictions
        preds_cls, preds_anc, preds_box = converter(predictions)
    
    # Convert predictions to the format expected by draw_bboxes
    boxes = convert_predictions_to_boxes(preds_cls, preds_box, args.conf_thres)
    
    # Convert OpenCV image to PIL Image for drawing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Draw results
    pil_image = draw_bboxes(pil_image, [boxes], idx2label=cfg.dataset.class_list)
    
    # Convert back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return frame, boxes

@hydra.main(config_path="../yolo/config", config_name="config", version_base=None)
def main(cfg: Config):
    args = parse_args()
    model, cfg, class_names = create_model_and_config(cfg, args)
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, cfg.device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    
    # Get webcam properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize performance metrics
    frame_count = 0
    total_time = 0
    
    print(f"Starting webcam feed at {width}x{height} @ {fps:.1f} FPS")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame and measure time
        start_time = time.time()
        frame, predictions = process_frame(frame, model, converter, cfg, args)
        inference_time = time.time() - start_time
        
        # Update performance metrics
        frame_count += 1
        total_time += inference_time
        
        # Calculate and display FPS
        if frame_count % 30 == 0:  # Update FPS display every 30 frames
            avg_fps = 1.0 / (total_time / frame_count)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLO Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Average FPS: {1.0 / (total_time / frame_count):.1f}")
    print(f"Average inference time: {(total_time / frame_count) * 1000:.1f}ms")

if __name__ == "__main__":
    main() 