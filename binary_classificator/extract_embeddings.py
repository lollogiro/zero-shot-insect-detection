# script partially copied by https://github.com/AI-EcoNet/zeroshot_clustering_animal_images/blob/main/scripts/extract_embeddings.py

import os
import json
import random
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
import open_clip
from collections import defaultdict

from utils.config import DEFAULT_MODEL_CACHE_DIR

DATASETS_PATH = os.environ.get("ZERO_SHOT_DATA_DIR", "./data")

# ANNOTATIONS_FILE = f"{DATASETS_PATH}/pitfall-cameras/annotations/pitfall-cameras_test_unified.json"
# IMAGES_DIR = f"{DATASETS_PATH}/pitfall-cameras/images"

# ANNOTATIONS_FILE = f"{DATASETS_PATH}/7395752/test1201_coco.json"
# IMAGES_DIR = f"{DATASETS_PATH}/7395752/test1201"

# ANNOTATIONS_FILE = f"{DATASETS_PATH}/greenland_test_data/coco_region_data.json"
# IMAGES_DIR = f"{DATASETS_PATH}/greenland_test_data/Annotated files"

ANNOTATIONS_FILE = f"{DATASETS_PATH}/stefan_smartphone/annotations/cropped/processed/ground_truth_coco.json"
IMAGES_DIR = f"{DATASETS_PATH}/stefan_smartphone/cropped/"

# ANNOTATIONS_FILE = f"{DATASETS_PATH}/rangex/detections_coco.json"
# IMAGES_DIR = f"{DATASETS_PATH}/rangex"

MODEL_NAME = "bioclip2_5" # "dinov2", "dinov3", "bioclip2", "bioclip2_5", "eupe_vitb16"
OUTPUT_FILE = f"extracted_embeddings/stefan_test_embeddings_{MODEL_NAME}.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    if x1 + w1 <= x2 or x2 + w2 <= x1: return False
    if y1 + h1 <= y2 or y2 + h2 <= y1: return False
    return True

def get_random_background_crop(img_w, img_h, crop_w, crop_h, all_gt_boxes, max_retries=50):
    for _ in range(max_retries):
        max_x = max(0, img_w - crop_w)
        max_y = max(0, img_h - crop_h)
        
        rx = random.randint(0, int(max_x))
        ry = random.randint(0, int(max_y))
        
        candidate_box = [rx, ry, crop_w, crop_h]
        
        overlap = False
        for gt_box in all_gt_boxes:
            if check_overlap(candidate_box, gt_box):
                overlap = True
                break
                
        if not overlap:
            return candidate_box
            
    return None

def load_feature_extractor(model_choice, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.environ.get("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
    if model_choice == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        model = model.to(DEVICE).eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def extract(img):
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                return model(tensor).squeeze(0).cpu()
        return extract

    elif model_choice == "dinov3":
        model_id = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'

        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = model.to(DEVICE).eval()

        def extract(img):
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # extracting the CLS token from the last hidden state
                return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return extract

    elif model_choice == "bioclip2":
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2', cache_dir=cache_dir)
        model = model.to(DEVICE).eval()

        def extract(img):
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                return model.encode_image(tensor).squeeze(0).cpu()
        return extract

    elif model_choice == "bioclip2_5":
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14', cache_dir=cache_dir)
        model = model.to(DEVICE).eval()

        def extract(img):
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                return model.encode_image(tensor).squeeze(0).cpu()
        return extract
    elif model_choice == "eupe_vitb16":
        EUPE_REPO_DIR = os.environ.get("EUPE_REPO_DIR", "./EUPE")
        model = torch.hub.load(EUPE_REPO_DIR, 'eupe_vitb16', source='local', weights=f"{cache_dir}/EUPE-ViT-B.pt")
        model = model.to(DEVICE).eval()
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        
        def extract(img):
            batch_img = transform(img).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                outputs = model.forward_features(batch_img)
                clstoken = outputs["x_norm_clstoken"].squeeze(0).cpu()
            return clstoken
        return extract
    else:
        raise ValueError("Invalid model choice. Use 'dinov2', 'dinov3', 'bioclip2', 'bioclip2_5', or 'eupe_vitb16'.")

def main():
    print(f"Loading {MODEL_NAME.upper()} onto {DEVICE}...")
    extract_embedding = load_feature_extractor(MODEL_NAME)
    
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco_data = json.load(f)
        
    img_info_map = {img['id']: img for img in coco_data['images']}
    
    anns_by_img = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_img[ann['image_id']].append(ann)
        
    all_embeddings = []
    all_labels = [] 
    
    for img_id, anns in tqdm(anns_by_img.items()):
        img_info = img_info_map[img_id]
        
        img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
            
        img_w, img_h = image.size
        gt_boxes = [ann['bbox'] for ann in anns]
        
        for ann in anns:
            x, y, w, h = [int(v) for v in ann['bbox']]
            
            if w <= 5 or h <= 5: 
                continue
                
            # insect crop (label 1)
            insect_crop = image.crop((x, y, x+w, y+h))
            emb_insect = extract_embedding(insect_crop)
            all_embeddings.append(emb_insect)
            all_labels.append(1)
            
            # background crop (label 0)
            bg_box = get_random_background_crop(img_w, img_h, w, h, gt_boxes)
            if bg_box:
                bx, by, bw, bh = bg_box
                bg_crop = image.crop((bx, by, bx+bw, by+bh))
                emb_bg = extract_embedding(bg_crop)
                all_embeddings.append(emb_bg)
                all_labels.append(0)

    X = torch.stack(all_embeddings)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"\nExtracted {len(y)} embeddings using {MODEL_NAME}.")
    print(f"Class distribution:\n\t{torch.sum(y).item()} insects\n\t{len(y) - torch.sum(y).item()} backgrounds.")
    
    torch.save({"features": X, "labels": y}, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()