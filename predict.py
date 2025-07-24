import os
import json
import torch
import numpy as np
import cv2
import argparse

from models.network_sr import SuperResolution
from utils.utils_image import tensor2uint, imsave
from utils.utils_chekpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model checkpoint')
    parser.add_argument('--folder_lq', type=str, required=True, help='Low-resolution image folder')
    parser.add_argument('--tile_size', type=int, default=0, help='Tile size for tile-based inference (0 = disable)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap between tiles (default=32)')

    args = parser.parse_args()
    
    config_path, model_path, folder_lq = args.config, args.model_path, args.folder_lq
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    scale = config['scale']
    model_cfg = config['model']
    num_layer = model_cfg['num_layer']
    embed_dim = model_cfg['embed_dim']
    act = model_cfg['act']
    resi = '1conv' if model_cfg['resi'] else '3conv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SuperResolution(
        scale=scale,
        num_layer=num_layer,
        embed_dim=embed_dim,
        act=act,
        resi=resi
    ).to(device)

    model, _, _, _, _ = load_checkpoint(model, None, None, model_path)
    model.eval()

    lq_images = sorted(os.listdir(folder_lq))
    folder_name = os.path.basename(os.path.normpath(folder_lq))
    save_dir = os.path.join("predict", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving results to: {save_dir}")
    with torch.no_grad():
        for fname in lq_images:
            lq_path = os.path.join(folder_lq, fname)
            lq_tensor, _ = load_image(lq_path)
            lq_tensor = lq_tensor.to(device)

            if args.tile_size > 0:
                output = tile_based_inference(model, lq_tensor, args.tile_size, args.tile_overlap, scale)
            else:
                output = model(lq_tensor).clamp(0, 1)
            sr = tensor2uint(output)

            save_img_path = os.path.join(save_dir, fname)
            imsave(sr, save_img_path)
            print(f"Saved: {fname}")

    print("[INFO] Inference completed.")
    

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255. 
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return img_tensor, img


def tile_based_inference(model, lq_tensor, tile_size, tile_overlap, scale):
    b, c, h, w = lq_tensor.size()
    tile = tile_size
    overlap = tile_overlap
    stride = tile - overlap
    output = torch.zeros((b, c, h * scale, w * scale), device=lq_tensor.device)
    weight_map = torch.zeros_like(output)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)
            y_start = max(y_end - tile, 0)
            x_start = max(x_end - tile, 0)

            lq_patch = lq_tensor[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                sr_patch = model(lq_patch).clamp(0, 1)

            out_y_start = y_start * scale
            out_y_end = y_end * scale
            out_x_start = x_start * scale
            out_x_end = x_end * scale

            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += sr_patch
            weight_map[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1.0

    output /= weight_map
    return output


if __name__ == '__main__':
    main()
