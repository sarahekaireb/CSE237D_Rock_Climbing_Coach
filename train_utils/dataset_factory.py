from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision import transforms

import os
import json
from pycocotools.coco import COCO
# https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4

class COCO_HoldDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, annotations):
        self.data_dir = dataset_dir
        self.annotations = annotations
        
        self.id_paths = []
        for img in self.annotations['images']:
            self.id_paths.append(img['file_name'])
        
        self.bboxes = {} # each image id has list of self.bboxes (each bbox is a list of 4 elems, xmin, ymin, w, h)
        self.labels = {}
        self.areas = {}
        self.crowds = {}
        for annot in self.annotations['annotations']:
            if annot['image_id'] not in self.bboxes:
                self.bboxes[annot['image_id']] = []
            if annot['image_id'] not in self.labels:
                self.labels[annot['image_id']] = []
            if annot['image_id'] not in self.areas:
                self.areas[annot['image_id']] = []
            if annot['image_id'] not in self.crowds:
                self.crowds[annot['image_id']] = []

            xmin, ymin, w, h = annot['bbox']
            xmax, ymax = xmin+w, ymin+h
            self.bboxes[annot['image_id']].append([xmin, ymin, xmax, ymax])
            self.labels[annot['image_id']].append(annot['category_id'])
            self.areas[annot['image_id']].append(annot['area'])
            self.crowds[annot['image_id']].append(annot['iscrowd'])

        self.targets = {}


    def __getitem__(self, idx):
        img = F.pil_to_tensor(Image.open(os.path.join(self.data_dir, self.id_paths[idx]))).float()
        bboxes = torch.FloatTensor(self.bboxes[idx]) # N x 4
        label = torch.FloatTensor(self.labels[idx]) # N
        if not torch.cuda.is_available():
            target = {
                'boxes': torch.FloatTensor(self.bboxes[idx]),
                'labels': torch.FloatTensor(self.labels[idx]),
                'image_id': torch.tensor([idx]),
                'area': torch.FloatTensor(self.areas[idx]),
                'iscrowd': torch.tensor(self.crowds[idx])
            }
        else:
            target = {
                'boxes': torch.FloatTensor(self.bboxes[idx]).cuda(),
                'labels': torch.LongTensor(self.labels[idx]).cuda(),
                'image_id': torch.tensor([idx]).cuda(),
                'area': torch.FloatTensor(self.areas[idx]).cuda(),
                'iscrowd': torch.tensor(self.crowds[idx]).cuda()
            }

        return img, target

    def __len__(self):
        return len(self.id_paths)
        

class COCO_WallDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, train_type):
        """
        reference: https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047

        Creates a torch Dataset for use in a DataLoader

        dataset_dir: this folder should contain three subfolders (train, valid, test)
                        each subfolder should have images as well as _annotations.coco.json files
        train_type: one of {train, valid, test}
        """
        self.train_type = train_type
        self.data_dir = os.path.join(dataset_dir, self.train_type)
        self.coco = COCO(os.path.join(self.data_dir, '_annotations.coco.json'))
        
        self.orig_images, self.orig_masks = self._load_data()
        if train_type != 'test':
            self.images, self.masks = self._augment_dataset()
        else:
            self.images = self.orig_images
            self.masks = self.orig_masks

    def _load_data(self):
        print("Loading Images")
        images = []
        masks = []
        for i in range(len(self.coco.getImgIds())):
            img_metadata = self.coco.loadImgs(i)[0]
            img_fn = img_metadata['file_name']
            img = np.array(Image.open(os.path.join(self.data_dir, img_fn)))
            img = torch.LongTensor(img).permute(2, 0, 1)
            img = self.preprocess(img)

            ann_id = self.coco.getAnnIds(i)[0]
            ann = self.coco.loadAnns(ann_id)[0]
            mask = torch.LongTensor(self.coco.annToMask(ann))

            images.append(img)
            masks.append(mask)

        return images, masks

    def __len__(self):
        # return len(self.coco.getImgIds())
        return len(self.images)

    def _augment_dataset(self):
        images = []
        masks = []
        print("Creating Augmentations")
        for i in range(len(self.orig_images)):
            img, mask = self.orig_images[i], self.orig_masks[i]
            # construct new images
            new_images, new_masks = self.augment_image(img, mask)
            for j in range(len(new_images)):
                images.append(new_images[j])
                masks.append(new_masks[j])
            print("Augmented image {} for {} new imgs".format(i, len(new_images)))
        
        return images, masks
    
    def augment_image(self, img, mask):
        # returns list of images, each one being some augmentation of the original
        img1, img2, img3, img4, img5 = F.five_crop(img, size=(162, 115))
        mask1, mask2, mask3, mask4, mask5 = F.five_crop(mask, size=(162, 115))

        cj = transforms.ColorJitter(brightness=0.5, hue=0.3, saturation=0.2, contrast=0.4)
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        gray = transforms.Grayscale(num_output_channels=3)

        aug_imgs = [norm(cj(im)) for im in [img1, img2, img3, img4, img5]] + [norm(gray(im)) for im in [img1, img2, img3, img4, img5]]
        aug_masks = [mask1, mask2, mask3, mask4, mask5] * 2
        flipped_imgs = [F.vflip(im) for im in aug_imgs] + [F.hflip(im) for im in aug_imgs]
        flipped_masks = [F.vflip(mask) for mask in aug_masks] + [F.hflip(mask) for mask in aug_masks]

        return aug_imgs + flipped_imgs, aug_masks + flipped_masks
    
    def preprocess(self, img):
        # img is 3 x H x W
        img_min = img.flatten(start_dim=1).min(dim=1).values.view(-1, 1, 1)
        img_max = img.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1)

        minmax_normed = (img - img_min) / (img_max - img_min)
        # normalized = F.normalize(minmax_normed, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return minmax_normed

    def __getitem__(self, idx):
        # img id is idx
        # img_metadata = self.coco.loadImgs(idx)[0]
        # img_fn = img_metadata['file_name']
        # img = np.array(Image.open(os.path.join(self.data_dir, img_fn)))
        # img = torch.LongTensor(img).permute(2, 0, 1)
        # img = self.preprocess(img)

        # ann_id = self.coco.getAnnIds(idx)[0]
        # ann = self.coco.loadAnns(ann_id)[0]
        # mask = self.coco.annToMask(ann)

        # return img, torch.LongTensor(mask)
        return self.images[idx], self.masks[idx]


def get_hold_dataset(data_dir, train_type):
    assert train_type in ['train', 'test', 'valid']
    dataset_dir = os.path.join(data_dir, train_type)
    with open(os.path.join(dataset_dir, '_annotations.coco.json'), 'r') as f:
        annotations = json.load(f)

    dataset = COCO_HoldDataset(dataset_dir, annotations)
    return dataset

def get_wall_dataset(data_dir, train_type):
    assert train_type in ['train', 'test', 'valid']
    
    dataset = COCO_WallDataset(data_dir, train_type)
    return dataset

# if __name__ == '__main__':
#     # Testing functionality
#     temp = get_dataset('datasets/orig_dataset_coco', 'train')
#     print(len(temp))
#     print(temp.__getitem__(3)[0].shape)