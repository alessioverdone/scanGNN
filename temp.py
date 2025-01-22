from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob

import numpy as np
from config import Configuration
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset, RandomSampler
from torchvision import transforms
import random


def read_mimic_(batchsize, data_dir='../mimic_part_jpg'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5, 5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: dataset_creation_2(mode=x, transform=data_transforms[x])
                      for x in ['train', 'test']}

    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['test'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )

    return data_loader_train, data_loader_test





def read_mimic(batchsize, data_dir='../mimic_part_jpg'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5, 5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: dataset(mode=x, transform=data_transforms[x])
                      for x in ['train', 'test']}

    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['test'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )

    return data_loader_train, data_loader_test



# per passare img e fixmaps separate
class Dataset_Union_Images_Scanpath_2(Dataset):

    def __init__(self, directory_scanpath, directory_img, directory_fixpoints, transform_img=None,
                 transform_scanpath=None):
        # def path
        self.directory_scanpath = directory_scanpath
        self.directory_img = directory_img
        self.directory_fixpoints = directory_fixpoints

        # carico dataset
        self.dataset_scanpath = self.load_data_scanpath()
        self.dataset_images, self.dataset_fixpoints = self.load_data_img()
        self.dataset_labels = self.load_labels()

        # initialize transform
        self.transform_img = transform_img
        self.transform_scanpath = transform_scanpath

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data_scanpath(self):
        scanpath = []
        path = sorted(os.listdir(self.directory_scanpath),
                      key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))
        for i in path:
            file_path = os.path.join(self.directory_scanpath, i)
            # print(file_path)
            loaded_array = np.load(file_path)
            scanpath.append(loaded_array.astype(float))
        # longest_sequence_lenght = max(scanpath, key=lambda x: x.shape[0]).shape[0]
        longest_sequence_lenght = 1000  # max lenghts are 33 and 25
        scanpath_torch = torch.zeros((len(scanpath), longest_sequence_lenght, 3), dtype=torch.float)
        for id, elem in enumerate(scanpath):
            elem_torch = torch.from_numpy(elem).float()
            scanpath_torch[id, :len(elem_torch), :] = elem_torch
        return scanpath_torch

    def load_labels(self):
        labels = []
        if "ASD" in self.directory_scanpath:
            labels = np.ones(len(self.dataset_scanpath))
        else:
            labels = np.zeros(len(self.dataset_scanpath))

            # deve avere shape [1,1]
        labels = labels.reshape(-1, 1)
        # devono essere float32
        labels = labels.astype(np.float32)

        return labels

    def load_data_img(self):
        images = []
        fixpoints = []

        img_path = sorted(os.listdir(self.directory_img),
                          key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))
        fixpoints_path = sorted(os.listdir(self.directory_fixpoints),
                                key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))

        for i, j in zip(img_path, fixpoints_path):
            images.append(os.path.join(self.directory_img, i))
            fixpoints.append(os.path.join(self.directory_fixpoints, j))

        return images, fixpoints

    def __len__(self):
        return len(self.dataset_scanpath)

    def __getitem__(self, index):
        labels = self.dataset_labels[index]
        scanpath = self.dataset_scanpath[index]
        # scanpath = self.dataset_scanpath[index].astype(np.float32)
        # if len(scanpath) < 3:
        #     print('scanpath: ', scanpath)

        # if self.transform_scanpath:
        # scanpath = self.transform_scanpath(scanpath)
        # scanpath = torch.from_numpy(scanpath)

        img_elem = self.dataset_images[index]
        fixpoints_elem = self.dataset_fixpoints[index]

        with Image.open(img_elem) as img, Image.open(fixpoints_elem) as fixpoints_img:
            if self.transform_img:
                img = self.transform_img(img)  # (1024,756)
                fixpoints_img = self.transform_img(fixpoints_img)

                # print('img shape: ', img.shape)
                # print('fixpoints shape: ', fixpoints_img.shape)

            # per concatenare img e fixpoints sulla dimensione giusta
            img = img.permute(0, 2, 1)
            fixpoints_img = fixpoints_img.permute(0, 2, 1)
            # concat_image = torch.cat((fixpoints_img, img), 0)

            # Sposta i tensori sul dispositivo corretto
            scanpath = scanpath.to(self.device)
            original_image = img.to(self.device)
            fixpoints_image = fixpoints_img.to(self.device)
            labels = torch.tensor(labels, device=self.device)

        return (scanpath, original_image, fixpoints_image), labels


def dataset_creation_2(params, transform_img_resize, transform_scanpath, mode=None, transform=None):
    datasetASD = Dataset_Union_Images_Scanpath_2(params.directory_scanpath_ASD,
                                                 params.directory_img_ASD,
                                                 params.directory_fixpoints_ASD,
                                                 transform_img=transform_img_resize,
                                                 transform_scanpath=transform_scanpath)
    datasetTD = Dataset_Union_Images_Scanpath_2(params.directory_scanpath_TD,
                                                params.directory_img_TD,
                                                params.directory_fixMaps_TD,
                                                transform_img=transform_img_resize,
                                                transform_scanpath=transform_scanpath)

    dataset = ConcatDataset([datasetASD, datasetTD])
    return dataset




class dataset(Dataset):

    def __init__(self, data_dir='../mimic_part_jpg', mode="train", transform=None):

        self.root = data_dir
        self.mode = mode
        self.T = transform
        self.csv = pd.read_csv(os.path.join(self.root, "gaze", "fixations.csv"))
        self.labels = ["CHF", "Normal", "pneumonia"]
        self.labelsdict = {"CHF": 0, "Normal": 1, "pneumonia": 2}
        self.idlist = []
        for i in range(len(self.labels)):
            self.idlist.extend(glob(os.path.join(self.root, self.mode, self.labels[i], "*.jpg")))

    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):

        # get path
        imgpath = self.idlist[idx]
        id = imgpath.split("/")[-1].split(".jpg")[0]
        gazepath = os.path.join(self.root, "gaze", "fixations", "{}.npy".format(id))

        # extract image
        with open(imgpath, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # extract label
        # label = self.labelsdict[imgpath.split("/")[-2]]
        label = self.labelsdict[imgpath.split("\\")[-2]]

        # extract gaze
        id = imgpath.split("\\")[-1].split(".jpg")[0]
        gaze = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
        idcsv = self.csv.loc[self.csv["DICOM_ID"] == id]
        for i in range(len(idcsv)):
            if i == 0:
                t = idcsv.iloc[i]["Time (in secs)"]
            else:
                t = idcsv.iloc[i]["Time (in secs)"] - idcsv.iloc[i - 1]["Time (in secs)"]
            x = idcsv.iloc[i]["X_ORIGINAL"]
            y = idcsv.iloc[i]["Y_ORIGINAL"]
            gaze[y, x] = t
        gaze = np.log(gaze + 0.01)
        gaze = (((gaze - gaze.min()) / (gaze.max() - gaze.min())) * 255).astype(np.uint8)
        gimg = gaze[..., np.newaxis].repeat(3, axis=2)
        gimg = Image.fromarray(gimg)

        # transform
        state = torch.get_rng_state()
        img = self.T(img)
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        torch.set_rng_state(state)
        gaze = self.T(gimg)
        gaze = self.getPatchGaze(gaze[0])

        return img, label, gaze

    def getPatchGaze(self, gaze):
        g = np.zeros((56, 56), dtype=np.float32)
        for i in range(56):
            for j in range(56):
                x1 = 4 * i - 7
                x2 = 4 * i + 7
                y1 = 4 * j - 7
                y2 = 4 * j + 7
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > 223:
                    x2 = 223
                if y2 > 223:
                    y2 = 223
                g[i, j] = gaze[x1:x2, y1:y2].sum()
        if g.max() - g.min() != 0:
            g = (g - g.min()) / (g.max() - g.min())
        return g

