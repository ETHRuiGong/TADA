import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *


class cityscapesLoader_coarsetofine_fewshot_numsample_classspecific_full(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=False,
        augmentations=None,
        version="cityscapes",
        return_id=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876]),
        return_annotated = True,
        num_fewshotsamples = None
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = img_mean
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit_trainvaltest","leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine_trainvaltest", "gtFine", self.split
        )
        self.num_samples = num_fewshotsamples

        # self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        if split == 'train':


            with open("./process_data/vegetation_coarsetofine.txt", 'r') as vegetation_file:
                self.vegetation_file = vegetation_file.readlines()
            vegetation_index = list(range(len(self.vegetation_file)))
            with open("./process_data/terrain_coarsetofine.txt", 'r') as terrain_file:
                self.terrain_file = terrain_file.readlines()
            terrain_index = list(range(len(self.terrain_file)))
            with open("./process_data/person_coarsetofine.txt", 'r') as person_file:
                self.person_file = person_file.readlines()
            person_index = list(range(len(self.person_file)))
            with open("./process_data/rider_coarsetofine.txt", 'r') as rider_file:
                self.rider_file = rider_file.readlines()
            rider_index = list(range(len(self.rider_file)))
            with open("./process_data/motorcycle_coarsetofine.txt", 'r') as motorcycle_file:
                self.motorcycle_file = motorcycle_file.readlines()
            motorcycle_index = list(range(len(self.motorcycle_file)))
            with open("./process_data/bicycle_coarsetofine.txt", 'r') as bicycle_file:
                self.bicycle_file = bicycle_file.readlines()
            bicycle_index = list(range(len(self.bicycle_file)))

            print("Coarse to Fine !!!!!!!!!!!!!!!!!!")

            self.files[split] = []
            self.vegetation_selected = []
            self.terrain_selected = []
            self.person_selected = []
            self.rider_selected = []
            self.motorcycle_selected = []
            self.bicycle_selected = []
            for vegetation_id in vegetation_index:
                self.files[split].append(os.path.join(self.images_base, self.vegetation_file[vegetation_id].strip()))
                self.vegetation_selected.append(os.path.join(self.images_base, self.vegetation_file[vegetation_id].strip()))
            for terrain_id in terrain_index:
                self.files[split].append(os.path.join(self.images_base, self.terrain_file[terrain_id].strip()))
                self.terrain_selected.append(os.path.join(self.images_base, self.terrain_file[terrain_id].strip()))
            for person_id in person_index:
                self.files[split].append(os.path.join(self.images_base, self.person_file[person_id].strip()))
                self.person_selected.append(os.path.join(self.images_base, self.person_file[person_id].strip()))
            for rider_id in rider_index:
                self.files[split].append(os.path.join(self.images_base, self.rider_file[rider_id].strip()))
                self.rider_selected.append(os.path.join(self.images_base, self.rider_file[rider_id].strip()))
            for motorcycle_id in motorcycle_index:
                self.files[split].append(os.path.join(self.images_base, self.motorcycle_file[motorcycle_id].strip()))
                self.motorcycle_selected.append(os.path.join(self.images_base, self.motorcycle_file[motorcycle_id].strip()))
            for bicycle_id in bicycle_index:
                self.files[split].append(os.path.join(self.images_base, self.bicycle_file[bicycle_id].strip()))
                self.bicycle_selected.append(os.path.join(self.images_base, self.bicycle_file[bicycle_id].strip()))
            # print("self.files", self.files["train"])


        # else:
        #     self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        # self.annotated_classes = [3, 4, 5, 9, 14, 16]
        self.id_to_trainid_annotated = {8: 8, 9: 9, 11: 11, 12: 12, 17: 17, 18: 18}
        self.vegetation_id = {8:8}
        self.terrain_id = {9:9}
        self.person_id = {11:11}
        self.rider_id = {12:12}
        self.motorcycle_id = {17: 17}
        self.bicycle_id = {18:18}
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 255 # 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        self.return_id = return_id

        self.return_annotated = return_annotated

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2], # temporary for cross validation
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        lbl_annotated = self.ignore_index * np.ones(lbl.shape, dtype=np.uint8)
        id_classspecific = {}
        if img_path in self.vegetation_selected:
            id_classspecific.update(self.vegetation_id)
        if img_path in self.bicycle_selected:
            id_classspecific.update(self.bicycle_id)
        if img_path in self.person_selected:
            id_classspecific.update(self.person_id)
        if img_path in self.terrain_selected:
            id_classspecific.update(self.terrain_id)
        if img_path in self.motorcycle_selected:
            id_classspecific.update(self.motorcycle_id)
        if img_path in self.rider_selected:
            id_classspecific.update(self.rider_id)
        # else:
        #     raise Exception(
        #         "No class specific fiels Founded"
        #     )
        # print("id_classspecific....", id_classspecific)
        if not id_classspecific:
            raise Exception(
                    "No class specific fiels Founded"
                )
        for k, v in id_classspecific.items():
            lbl_annotated[lbl == k] = v

        img_name = img_path.split('/')[-1]
        if self.return_id:
            return img, lbl, img_name, img_name, index

        if self.return_annotated:
            return img, lbl, lbl_annotated, img_path, lbl_path, img_name

        return img, lbl, img_path, lbl_path, img_name

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

'''
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "./data/city_dataset/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
'''
