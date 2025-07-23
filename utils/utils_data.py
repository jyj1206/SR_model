import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import utils.utils_image as utils
import time

class PairedTransform:
    def __init__(self, config):
        self.to_tensor = transforms.ToTensor()
        self.mode = config['dataset']['mode']
        self.patch_size = config['dataset']['patch_size']
        self.resize_hr = config['dataset']['resize_hr']
        self.scale = config['scale']

        # Resize 이미지 저장 경로
        self.exp_dir = config['exp_dir']
        self.pre_hr_dir = os.path.join(self.exp_dir, 'preprocess_images', 'HR')
        self.pre_lr_dir = os.path.join(self.exp_dir, 'preprocess_images', 'LR')
        os.makedirs(self.pre_hr_dir, exist_ok=True)
        os.makedirs(self.pre_lr_dir, exist_ok=True)

    def __call__(self, h_img, l_img, phase='train', h_img_path=None):        
        if phase == 'train':
            # Resize 전체 이미지 (mode == 'resize')
            if self.mode == 'resize':
                assert h_img_path is not None, "h_img_path must be provided in 'resize' mode."
                base_name = os.path.splitext(os.path.basename(h_img_path))[0]
                hr_fname = f"{base_name}_HR{self.resize_hr}.png"
                lr_size = int(self.resize_hr // self.scale)
                lr_fname = f"{base_name}_LR{lr_size}.png"

                hr_path = os.path.join(self.pre_hr_dir, hr_fname)
                lr_path = os.path.join(self.pre_lr_dir, lr_fname)

                if os.path.exists(hr_path) and os.path.exists(lr_path):
                    h_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
                    l_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                else:
                    h_img = cv2.resize(h_img, (self.resize_hr, self.resize_hr), interpolation=cv2.INTER_CUBIC)
                    l_size = int(self.resize_hr // self.scale)
                    l_img = cv2.resize(l_img, (l_size, l_size), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(hr_path, h_img)
                    cv2.imwrite(lr_path, l_img)

            # Patch 추출 (mode == 'patch')
            if self.mode == 'patch':
                H, W, _ = l_img.shape
                L_size = self.patch_size // self.scale

                if H < L_size or W < L_size:
                    raise ValueError(f"LR image size too small for patch cropping: ({H}, {W}) < ({L_size}, {L_size})")

                rnd_h = np.random.randint(0, H - L_size + 1)
                rnd_w = np.random.randint(0, W - L_size + 1)
                l_img = l_img[rnd_h:rnd_h + L_size, rnd_w:rnd_w + L_size, :]
                h_img = h_img[rnd_h * self.scale:rnd_h * self.scale + self.patch_size,
                              rnd_w * self.scale:rnd_w * self.scale + self.patch_size, :]

            # 좌우 반전 (augmentation)
            if np.random.rand() < 0.5:
                h_img = cv2.flip(h_img, 1)
                l_img = cv2.flip(l_img, 1)

        # BGR to RGB 변환
        h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB)
        l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)

        return self.to_tensor(h_img), self.to_tensor(l_img)


class DatasetSR(Dataset):
    def __init__(self, H_folder_path, L_folder_path, transform, phase):
        super().__init__()
        self.h_list = self.get_image_path_list(H_folder_path)
        self.l_list = self.get_image_path_list(L_folder_path)
        self.transform = transform
        self.phase = phase

        if len(self.h_list) != len(self.l_list):
            raise ValueError("Mismatch in number of high-res and low-res images.")

    def __len__(self):
        return len(self.h_list)

    def __getitem__(self, index):
        h_path = self.h_list[index]
        l_path = self.l_list[index]
        
        h_img = cv2.imread(h_path, cv2.IMREAD_COLOR)
        l_img = cv2.imread(l_path, cv2.IMREAD_COLOR)

        h_img, l_img = self.transform(h_img, l_img, self.phase, h_img_path=h_path)

        return {'L': l_img, 'H': h_img, 'L_path': l_path, 'H_path': h_path}

    def get_image_path_list(self, path):
        img_path_list = []
        for dirpath, _, fnames in os.walk(path):
            for fname in sorted(fnames):
                if utils.is_image_file(fname): 
                    img_path = os.path.join(dirpath, fname)
                    img_path_list.append(img_path)
        return img_path_list


def get_dataloader(phase, dataset, batch_size=64, num_workers=8):
    if phase == 'train':
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        return train_loader

    elif phase == 'test':
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return test_loader

    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'train' or 'test'.")
