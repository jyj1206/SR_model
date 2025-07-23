import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import utils.utils_image as utils


class PairedTransform:
    def __init__(self, config):
        self.to_tensor = transforms.ToTensor()
        self.patch_size = config['dataset']['patch_size']
        self.scale = config['scale']

        # Resize 이미지 저장 경로
        self.exp_dir = config['exp_dir']
        self.pre_hr_dir = os.path.join(self.exp_dir, 'preprocess_images', 'HR')
        self.pre_lr_dir = os.path.join(self.exp_dir, 'preprocess_images', 'LR')
        os.makedirs(self.pre_hr_dir, exist_ok=True)
        os.makedirs(self.pre_lr_dir, exist_ok=True)

    def __call__(self, h_img, l_img, phase='train', h_img_path=None):
        if phase == 'train':
            H, W, _ = l_img.shape
            L_size = self.patch_size // self.scale

            if H < L_size or W < L_size:
                raise ValueError(f"LR image size too small for patch cropping: ({H}, {W}) < ({L_size}, {L_size})")

            # 랜덤 크롭
            rnd_h = np.random.randint(0, H - L_size + 1)
            rnd_w = np.random.randint(0, W - L_size + 1)
            l_img = l_img[rnd_h:rnd_h + L_size, rnd_w:rnd_w + L_size, :]
            h_img = h_img[rnd_h * self.scale:rnd_h * self.scale + self.patch_size,
                          rnd_w * self.scale:rnd_w * self.scale + self.patch_size, :]

            # augmentation 
            mode = np.random.randint(0, 8)
            l_img = utils.augment_img(l_img, mode)
            h_img = utils.augment_img(h_img, mode)

        # BGR → RGB
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
