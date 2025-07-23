import os
import argparse
import json
import torch
import numpy as np
import cv2

from models.network_sr import SuperResolution
from utils.utils_image import calculate_psnr, ssim, bgr2ycbcr, tensor2uint
from utils.utils_chekpoint import load_checkpoint
from torchsummary import summary

def load_image_pair(lq_path, gt_path):
    lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    lq_tensor = torch.from_numpy(lq.transpose(2, 0, 1)).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float()
    return lq_tensor, gt_tensor, lq, gt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model checkpoint')
    parser.add_argument('--folder_lq', type=str, required=True, help='Low-resolution image folder')
    parser.add_argument('--folder_gt', type=str, required=True, help='Ground-truth image folder')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    scale = config['scale']
    model_cfg = config['model']
    num_layer = model_cfg['num_layer']
    embed_dim = model_cfg['embed_dim']
    act = model_cfg['act']
    resi = '1conv' if model_cfg['resi'] else '3conv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = SuperResolution(
        scale=scale,
        num_layer=num_layer,
        embed_dim=embed_dim,
        act=act,
        resi=resi
    ).to(device)

    summary(model, input_size=(3, 128, 128))

    model, _, _, _, _ = load_checkpoint(model, None, None, args.model_path)
    model.eval()

    lq_images = sorted(os.listdir(args.folder_lq))
    psnr_total = psnr_y_total = ssim_total = ssim_y_total = 0.0
    count = 0

    with torch.no_grad():
        for fname in lq_images:
            lq_path = os.path.join(args.folder_lq, fname)
            gt_path = os.path.join(args.folder_gt, fname)
            if not os.path.exists(gt_path):
                print(f"GT not found for {fname}, skipping...")
                continue

            lq_tensor, gt_tensor, _, _ = load_image_pair(lq_path, gt_path)
            lq_tensor, gt_tensor = lq_tensor.to(device), gt_tensor.to(device)

            output = model(lq_tensor).clamp(0, 1)

            sr = tensor2uint(output)
            gt = tensor2uint(gt_tensor)

            psnr = calculate_psnr(sr, gt, border=scale)
            ssim_val = ssim(sr, gt)

            sr_y = bgr2ycbcr(sr.astype(np.float32) / 255.) * 255.
            gt_y = bgr2ycbcr(gt.astype(np.float32) / 255.) * 255.
            psnr_y = calculate_psnr(sr_y, gt_y, border=scale)
            ssim_y = ssim(sr_y, gt_y)

            print(f"{fname} | PSNR: {psnr:.2f} | PSNR_Y: {psnr_y:.2f} | SSIM: {ssim_val:.4f} | SSIM_Y: {ssim_y:.4f}")

            psnr_total += psnr
            psnr_y_total += psnr_y
            ssim_total += ssim_val
            ssim_y_total += ssim_y
            count += 1

    print("\n=== Evaluation Summary ===")
    print(f"Average PSNR:   {psnr_total / count:.2f} dB")
    print(f"Average PSNR_Y: {psnr_y_total / count:.2f} dB")
    print(f"Average SSIM:   {ssim_total / count:.4f}")
    print(f"Average SSIM_Y: {ssim_y_total / count:.4f}")

if __name__ == '__main__':
    main()
