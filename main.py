import os
import json
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from utils.utils_seed import set_seed
from utils.utils_logger import setup_logger, log_model_summary
from utils.utils_directory import get_next_exp_dir
from utils.utils_data import PairedTransform, DatasetSR, get_dataloader
from utils.utils_image import calculate_psnr, bgr2ycbcr, imsave, tensor2uint
from utils.utils_chekpoint import load_checkpoint, save_checkpoint

from models.network_sr import SuperResolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    args = parser.parse_args()

    '''
        option initialization
    '''
    with open(args.config, 'r') as f:
        config = json.load(f)

    h_path = config['train']['h_path']
    l_path = config['train']['l_path']
    test_h_path = config['test']['h_path']
    test_l_path = config['test']['l_path']
    scale = config['scale']
    seed = config['seed']
    resume_path = config.get('resume', None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = get_next_exp_dir()
    model_dir = os.path.join(exp_dir, "models")
    image_dir = os.path.join(exp_dir, "images")

    config_save_path = os.path.join(exp_dir, 'config.json')
    shutil.copy(args.config, config_save_path)

    """
        logger & seed setting
    """
    set_seed(seed)
    logger = setup_logger(log_dir=exp_dir, log_name=f"train.log")

    logger.info(f"High-res image path: {h_path} | Low-res image path: {l_path}")
    logger.info(f"Seed = {seed} | Device: {device}")
    logger.info("Training started")

    '''
        dataset, dataloader initialization
    '''
    config['exp_dir'] = exp_dir 
    transform = PairedTransform(config)

    train_set = DatasetSR(h_path, l_path, transform, 'train')
    test_set = DatasetSR(test_h_path, test_l_path, transform, 'test')

    train_loader = get_dataloader('train', train_set, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'])
    test_loader = get_dataloader('test', test_set, batch_size=1, num_workers=config['test']['num_workers'])

    '''
        model initialization
    '''
    num_layer = config['model']['num_layer']
    embed_dim = config['model']['embed_dim']
    act = config['model']['act']
    resi = config['model']['resi']

    model = SuperResolution(scale=scale, num_layer=num_layer, embed_dim=embed_dim, act=act, resi=resi).to(device)
    log_model_summary(model, input_size=(3, 512, 512), logger=logger)

    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    milestones = config['train']['milestones']
    scheduler_gamma = config['train']['scheduler_gamma']

    # Loss function
    criterion = nn.L1Loss()

    # Optimize
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma)

    # resume
    start_epoch = 0
    current_step = 0

    if resume_path:
        model, optimizer, scheduler, start_epoch, current_step = load_checkpoint(model, optimizer, scheduler, resume_path, device)
        logger.info(f"Resumed from checkpoint: {resume_path} (epoch {start_epoch}, step {current_step})")

    logger.info("Loss, optimizer, and scheduler are defined successfully.")

    '''
        main training
    '''
    epochs = config['train']['epochs']
    train_checkpoint_print = config['train']['train_checkpoint_print']
    save_checkpoint_print = config['train']['save_checkpoint_print']
    test_checkpoint_print = config['test']['test_checkpoint_print']

    model.train()
    for epoch in range(start_epoch, epochs):
        for train_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            current_step += 1

            H_img = train_data['H'].to(device)
            L_img = train_data['L'].to(device)

            optimizer.zero_grad()
            E_img = model(L_img)
            loss = criterion(E_img, H_img)
            loss.backward()
            optimizer.step()

            '''
                training information
            '''
            if current_step % train_checkpoint_print == 0:
                logger.info('<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, loss:{:.4f}>'.format(
                    epoch, current_step, scheduler.get_last_lr()[0], loss.item()
                ))

            '''
                save model
            '''
            if current_step % save_checkpoint_print == 0:
                ckpt_path = os.path.join(model_dir, f"epoch{epoch:03d}_step{current_step:06d}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, current_step, ckpt_path)
                logger.info(f"Checkpoint saved to: {ckpt_path}")

            '''
                testing
            '''
            if current_step % test_checkpoint_print == 0:
                avg_psnr = 0.0
                avg_psnr_y = 0.0
                idx = 0

                model.eval()
                with torch.no_grad():
                    for test_data in test_loader:
                        idx += 1
                        H_img = test_data['H'].to(device)
                        L_img = test_data['L'].to(device)
                        L_path = test_data['L_path']

                        E_img = model(L_img)

                        E_np = tensor2uint(E_img)
                        H_np = tensor2uint(H_img)

                        # save image
                        image_name_ext = os.path.basename(L_path[0])
                        img_name, _ = os.path.splitext(image_name_ext)

                        save_name = f"{img_name}_{current_step}.png"
                        save_path = os.path.join(image_dir, save_name)
                        imsave(E_np, save_path)

                        # PSNR (RGB)
                        psnr = calculate_psnr(E_np, H_np, border=scale)

                        # PSNR-Y
                        E_y = bgr2ycbcr(E_np.astype(np.float32) / 255.) * 255.
                        H_y = bgr2ycbcr(H_np.astype(np.float32) / 255.) * 255.
                        psnr_y = calculate_psnr(E_y, H_y, border=scale)

                        logger.info(f'{idx:>4d} --> {save_name} | PSNR: {psnr:.2f}dB | PSNR_Y: {psnr_y:.2f}dB')

                        avg_psnr += psnr
                        avg_psnr_y += psnr_y

                avg_psnr /= idx
                avg_psnr_y /= idx

                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR: {:.2f} dB | Average PSNR_Y: {:.2f} dB>\n'.format(
                    epoch, current_step, avg_psnr, avg_psnr_y
                ))
                model.train()

if __name__ == '__main__':
    main()
