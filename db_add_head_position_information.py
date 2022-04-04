import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.io import read_image
from deep_head_pose import hopenet, utils
from torch.utils.data import Dataset
from db_sqlAlchemy import session, HeadPositionTrain, HeadPositionValidation, HeadPositionTest
from parameters import CUDA_N

"""
pip install opencv-python==4.5.5.64
pip install torch torchvision
pip install scipy
"""

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]
trf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds)
])


def imshow(imgs, title=None):
    # imshow for img(tensor)
    imgs = imgs.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imgs = std * imgs + mean
    imgs = np.clip(imgs, 0, 1)
    plt.imshow(imgs)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class DatasetOneImage(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, x):
        # open image here as PIL / numpy
        image = read_image(self.image_path)
        label = [1]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def head_position_estimation_test_db():
    img_dir = '../data/val_set/images'
    print("Cuda available: ", torch.cuda.is_available())
    device = torch.device(CUDA_N if torch.cuda.is_available() else 'cpu')
    tbl = session.query(HeadPositionValidation).filter(HeadPositionValidation.camera_label == None).all()
    model = init_model_head_position(device)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    obj_lst = []
    for i, r in enumerate(tbl):
        img_path = os.path.join(img_dir, r.filename)
        dataset = DatasetOneImage(img_path, trf)
        dataloader = torch.utils.data.DataLoader(dataset)
        img, _ = next(iter(dataloader))
        camera_position = head_pose_est(model, img, device, idx_tensor)
        r.camera_label = camera_position
        print(img_path, camera_position)
    session.add_all(obj_lst)
    session.commit()


def init_model_head_position(device):
    # model = hopenet.Hopenet()
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    saved_state_dict = torch.load('deep_head_pose/hopenet_robust_alpha1.pkl', map_location=device)
    model.load_state_dict(saved_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def head_pose_est(model, image, device, idx_tensor):
    model = model.to(device)
    image = image.to(device)

    yaw, pitch, roll = model(image)

    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
    roll_predicted = utils.softmax_temperature(roll.data, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

    img_angle = [(x.item(), y.item(), z.item(),) for x, y, z in zip(yaw_predicted, pitch_predicted, roll_predicted)]
    # ttl = []
    text = None
    control_angle_left_right = 10
    control_angle_up_down = 12
    for item in img_angle:
        # obj_lst = []
        if abs(item[0]) > abs(item[1]):
            if abs(item[0]) >= control_angle_left_right:
                if item[0] < 0:
                    text = "Camera Left (Looking Right)"
                else:
                    text = "Camera Right (Looking Left)"
                # ttl.append(text)
                continue
        if abs(item[0]) < abs(item[1]):
            if abs(item[1]) > control_angle_up_down:
                if item[1] < 0:
                    text = "Camera up (Looking Down)"
                else:
                    text = "Camera down (Looking up)"
                # ttl.append(text)
                continue
        if abs(item[0]) <= control_angle_left_right and abs(item[1]) <= control_angle_up_down:
            text = "Forward"
            # ttl.append(text)

    #     obj = session.query(HeadPositionValidation).get(sample_fname)
    #     obj.camera_label = text
    #     obj_lst.append(obj)
    #     # session.add(q)
    # session.add_all(obj_lst)
    # session.commit()
    # imshow(torchvision.utils.make_grid(images), ttl)
    return text


# def NN():
#     print("Cuda available: ", torch.cuda.is_available())
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     # model = hopenet.Hopenet()
#     model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
#
#     saved_state_dict = torch.load('deep_head_pose/hopenet_robust_alpha1.pkl', map_location=device)
#     model.load_state_dict(saved_state_dict, strict=False)
#     model.eval()
#     model.to(device)
#     print('Ready network loaded.')
#
#     idx_tensor = [idx for idx in range(66)]
#     idx_tensor = torch.FloatTensor(idx_tensor).to(device)
#
#     yaw_error = .0
#     pitch_error = .0
#     roll_error = .0
#
#     l1loss = torch.nn.L1Loss(size_average=False)
#
#     # test_loader = get_dataloader()
#     test_dataset, test_loader = ds_test_all()
#     sample_dir = test_loader.dataset.img_dir
#
#     for i, (images, labels) in enumerate(test_loader):
#         images = images.to(device)
#         sample_fname = test_loader.dataset.img_labels.filename[i]
#         yaw, pitch, roll = model(images)
#
#         # Binned predictions
#         _, yaw_bpred = torch.max(yaw.data, 1)
#         _, pitch_bpred = torch.max(pitch.data, 1)
#         _, roll_bpred = torch.max(roll.data, 1)
#
#         # Continuous predictions
#         yaw_predicted = utils.softmax_temperature(yaw.data, 1)
#         pitch_predicted = utils.softmax_temperature(pitch.data, 1)
#         roll_predicted = utils.softmax_temperature(roll.data, 1)
#
#         yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
#         pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
#         roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
#
#         img_angle = [(x.item(), y.item(), z.item(),) for x, y, z in zip(yaw_predicted, pitch_predicted, roll_predicted)]
#         ttl = []
#         control_angle_left_right = 10
#         control_angle_up_down = 12
#         for item in img_angle:
#             obj_lst = []
#             if abs(item[0]) > abs(item[1]):
#                 if abs(item[0]) >= control_angle_left_right:
#                     if item[0] < 0:
#                         text = "Camera Left (Looking Right)"
#                     else:
#                         text = "Camera Right (Looking Left)"
#                     ttl.append(text)
#                     # continue
#             if abs(item[0]) < abs(item[1]):
#                 if abs(item[1]) > control_angle_up_down:
#                     if item[1] < 0:
#                         text = "Camera up (Looking Down)"
#                     else:
#                         text = "Camera down (Looking up)"
#                     ttl.append(text)
#                     # continue
#             if abs(item[0]) <= control_angle_left_right and abs(item[1]) <= control_angle_up_down:
#                 text = "Forward"
#                 ttl.append(text)
#
#             obj = session.query(HeadPositionValidation).get(sample_fname)
#             obj.camera_label = text
#             obj_lst.append(obj)
#             # session.add(q)
#         session.add_all(obj_lst)
#         session.commit()
#         imshow(torchvision.utils.make_grid(images), ttl)
#     pass


if __name__ == '__main__':
    head_position_estimation_test_db()
    # NN()
    pass

