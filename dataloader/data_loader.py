from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import numpy as np
import torch
from PIL import Image as PILImage


def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not opt.no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.img_size = make_dataset(opt.img_file)
        self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
        if not self.opt.isTrain:
            # Nếu test thì nhân mask lên để đủ số lượng
            self.mask_paths = self.mask_paths * max(1, (self.img_size // self.mask_size))
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        img, img_path = self.load_img(index)
        mask = self.load_mask(index)
        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path

    def load_mask(self, index):
        """Load mask from pre-generated file"""
        mask_path = self.mask_paths[index % len(self.mask_paths)]
        mask_pil = PILImage.open(mask_path).convert('L')  # Load mask as grayscale
        mask_resized = mask_pil.resize(self.opt.fineSize, resample=PILImage.NEAREST)
        mask_np = np.array(mask_resized)
        mask_np = (mask_np <= 127).astype(np.float32)  # Binarize
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # Add channel dimension
        return mask


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(
        datasets,
        batch_size=opt.batchSize,
        shuffle=not opt.no_shuffle,
        num_workers=int(opt.nThreads)
    )
    return dataset
