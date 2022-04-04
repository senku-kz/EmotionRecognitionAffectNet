import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

from db_sqlAlchemy import HeadPositionTrain, HeadPositionValidation, session
from parameters import folder_dst_lst, db_sql_file, batch_size, LIMIT_DATASET, SPLIT_TRAIN_SET


class CustomImageDatasetFromSQLTrain(Dataset):
    def __init__(self, cammera_position, img_dir, transform=None, target_transform=None):
        con = sqlite3.connect(db_sql_file)
        if cammera_position == 'all':
            if LIMIT_DATASET is None:
                smtm = f'SELECT filename, class_idx from {HeadPositionTrain.__tablename__}'
            else:
                smtm = f'SELECT filename, class_idx from {HeadPositionTrain.__tablename__} limit {LIMIT_DATASET}'
        else:
            smtm = f'SELECT filename, class_label from {HeadPositionTrain.__tablename__} where camera_label = "{cammera_position}"'
        self.img_labels = pd.read_sql_query(smtm, con)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = 0 if self.img_labels.empty else np.sort(self.img_labels['class_idx'].unique()).tolist()
        # if self.img_labels.empty:
        #     print('error')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        self.image = image
        self.label = label
        return image, label


class CustomImageDatasetFromSQLValidation(Dataset):
    def __init__(self, cammera_position, img_dir, transform=None, target_transform=None):
        con = sqlite3.connect(db_sql_file)
        if cammera_position == 'all':
            if LIMIT_DATASET is None:
                smtm = f'SELECT filename, class_idx from {HeadPositionValidation.__tablename__}'
            else:
                smtm = f'SELECT filename, class_idx from {HeadPositionValidation.__tablename__} limit {LIMIT_DATASET}'
        else:
            # smtm = f'SELECT filename, class_label from {HeadPositionValidation.__tablename__} where camera_label = "{cammera_position}"'
            smtm = f'SELECT filename, class_idx from {HeadPositionValidation.__tablename__} where camera_label = "{cammera_position}"'
        self.img_labels = pd.read_sql_query(smtm, con)
        self.img_dir = img_dir
        # self.name = None
        self.transform = transform
        self.target_transform = target_transform

        # self.classes = 0 if self.img_labels.empty else np.sort(self.img_labels['class_idx'].unique())
        if self.img_labels.empty:
            self.classes = 0
        else:
            if 'class_idx' in self.img_labels:
                self.classes = np.sort(self.img_labels['class_idx'].unique())
            if 'class_label' in self.img_labels:
                self.classes = np.sort(self.img_labels['class_label'].unique())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        name = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        self.image = image
        self.label = label
        self.name = name
        return image, label


def get_camera_position(dataset, position):
    if dataset == 'train':
        # q = session.query(HeadPositionTrain).all()
        c = session.query(HeadPositionTrain).filter(HeadPositionTrain.camera_label == position).count()
        q = session.query(HeadPositionTrain.filename, HeadPositionTrain.class_label).filter(HeadPositionTrain.camera_label == position).all()
        print(f'Count of rows is {c}')
        print(q)
    elif dataset == 'validation':
        c = session.query(HeadPositionValidation).filter(HeadPositionValidation.camera_label == position).count()
        q = session.query(HeadPositionTrain.filename, HeadPositionTrain.class_label).filter(HeadPositionValidation.camera_label == position).all()
        print(f'Count of rows is {c}')
        print(q)
    return q


def ds_train_validation_all():
    # batch_size = 8
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    # Data Transformation and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'validation': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])
    }

    dataset_uri = {
        'train': '../data/train_set/images',
        'validation': '../data/val_set/images'
    }

    # dataset_train = CustomImageDatasetFromSQL(cammera_position='Forward', img_dir=dataset_uri['train'], transform=data_transforms['train'])
    dataset_train = CustomImageDatasetFromSQLTrain(cammera_position='all', img_dir=dataset_uri['train'], transform=data_transforms['train'])
    train_size = int(SPLIT_TRAIN_SET * len(dataset_train))
    test_size = len(dataset_train) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_train, [train_size, test_size])

    # train_dataset = data_transforms['train'](train_dataset)
    # test_dataset = data_transforms['test'](test_dataset)

    dataset = {
        # 'train': CustomImageDatasetFromSQL(cammera_position='Forward', img_dir=dataset_uri['train'], transform=data_transforms['train']),
        'train': train_dataset,
        # 'validation': CustomImageDatasetFromSQL(cammera_position='Forward', img_dir=dataset_uri['validation'], transform=data_transforms['validation']),
        'test': test_dataset
    }

    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
        # 'validation': torch.utils.data.DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=True),
    }

    # annotations_data = get_camera_position('train', 'Forward')

    # class_names = dataset['train'].classes
    print(f'Number of training examples: \t', len(dataset['train']))
    # print(f'Number of validation examples: \t', len(dataset['validation']))
    print(f'Number of testing examples: \t', len(dataset['test']))
    # print('Class labels: \t', class_names)

    r_dataset = {}
    r_dataloader = {}

    r_dataset['train'] = dataset['train']
    r_dataset['validation'] = dataset['test']
    r_dataloader['train'] = dataloader['train']
    r_dataloader['validation'] = dataloader['test']
    # return r_dataset_train, r_dataset_validation, r_dataloader_train, r_dataloader_validation, train_size, test_size
    return r_dataset, r_dataloader


def ds_test_all():
    return ds_test_cam(camera_position='all')


def ds_test_cam(camera_position='all'):
    # batch_size = 8
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    # Data Transformation and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'validation': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])
    }

    dataset_uri = {
        # 'train': '../data/train_set/images',
        'validation': '../data/val_set/images'
    }

    dataset = {
        # 'train': train_dataset,
        'validation': CustomImageDatasetFromSQLValidation(cammera_position=camera_position, img_dir=dataset_uri['validation'], transform=data_transforms['validation']),
        # 'test': test_dataset
    }

    if dataset['validation'].img_labels.empty:
        print('Validation Dataset is empty!')
        return False, False

    dataloader = {
        # 'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
        'validation': torch.utils.data.DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True),
        # 'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=True),
    }

    # print(f'Number of training examples: \t', len(dataset['train']))
    # print(f'Number of validation examples: \t', len(dataset['validation']))
    # print(f'Number of test examples: \t', len(dataset['validation']))
    # print(f'Number of testing examples: \t', len(dataset['test']))
    # print('Class labels: \t', class_names)
    # r_dataset_test = dataset['validation']
    # r_dataloader_test = dataloader['validation']
    return dataset['validation'], dataloader['validation']


if __name__ == '__main__':
    v_dataset = 'train'
    # v_dataset = 'validation'
    v_position = 'Forward'
    # ds_front = get_camera_position(v_dataset, v_position)
    # ds_train_validation_all()
    ds_test_all()
    # ds_test_cam(camera_position='all')
    # ds_test_cam(camera_position=v_position)
