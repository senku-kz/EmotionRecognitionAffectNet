import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from sklearn import decomposition
# from sklearn import manifold
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import copy
import time
import matplotlib.pyplot as plt
import numpy as np

from models.VGG import VGG
from models.CoAtNet import coatnet_0
from split_dataset import ds_train_validation_all, ds_test_cam

from parameters import batch_size, epoch_number, learning_rate, model_dst


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]
        if normalize:
            image = normalize_image(image)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')
    plt.show()


def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5):
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def data_processing(batch_size=4):
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    # Data Transformation and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'test': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])
    }

    # ----
    dataset_uri = {
        'train': '../train_class',
        'val': '../val_class',
        'test': '../test_class'
    }
    dataset = {
        'train': datasets.ImageFolder(dataset_uri['train'], data_transforms['train']),
        'val': datasets.ImageFolder(dataset_uri['val'], data_transforms['val']),
        'test': datasets.ImageFolder(dataset_uri['test'], data_transforms['test'])
    }

    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=1),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, num_workers=1),
        'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=1),
    }
    class_names = dataset['train'].classes

    print(f'Number of training examples: \t', len(dataset['train']))
    print(f'Number of validation examples: \t', len(dataset['val']))
    print(f'Number of testing examples: \t', len(dataset['test']))
    print('Class labels: \t', class_names)

    N_IMAGES = 25

    images, labels = zip(*[(image, label) for image, label in
                           [dataset['train'][i] for i in range(N_IMAGES)]])

    # classes = dataset['train'].classes

    plot_images(images, labels, class_names)
    return dataloader, class_names


def data_processing_val(batch_size=4):
    VALID_RATIO = 0.9

    pretrained_size = (224, 224)
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    # Data Transformation and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'test': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means,
                                 std=pretrained_stds)
        ])
    }

    dataset_uri = {
        'train': '../AffectNet/train_class',
        'test': '../AffectNet/test_class'
    }
    dataset = {
        'train': datasets.ImageFolder(dataset_uri['train'], data_transforms['train']),
        'test': datasets.ImageFolder(dataset_uri['test'], data_transforms['test'])
    }

    n_train_examples = int(len(dataset['train']) * VALID_RATIO)
    n_valid_examples = len(dataset['train']) - n_train_examples

    v1, v2 = data.random_split(dataset['train'], [n_train_examples, n_valid_examples])
    dataset['train'], dataset['val'] = v1, v2

    dataset['val'] = copy.deepcopy(dataset['val'])
    dataset['val'].dataset.transform = data_transforms['val']

    dataloader = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=1),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, num_workers=1),
        'test': torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, num_workers=1),
    }
    class_names = dataset['train'].dataset.classes

    print(f'Number of training examples: \t', len(dataset['train']))
    print(f'Number of validation examples: \t', len(dataset['val']))
    print(f'Number of testing examples: \t', len(dataset['test']))
    print('Class labels: \t', class_names)

    N_IMAGES = 25

    images, labels = zip(*[(image, label) for image, label in
                           [dataset['train'][i] for i in range(N_IMAGES)]])

    plot_images(images, labels, class_names)
    return dataset, dataloader, class_names


def training_the_model(model, dataset, dataloader, epoch_num=1, lr=5e-4):
    dataset_size_train = len(dataset['train'])
    dataset_size_validation = len(dataset['validation'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    #
    model = model.to(device)
    # model_dst = './models_trained'
    log_train_accuracy = []
    log_train_loss = []
    log_valid_accuracy = []
    log_valid_loss = []

    #
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # =================
    # FOUND_LR = 5e-4

    # optimizer = optim.Adam(params, lr=FOUND_LR)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # ===
    # EPOCHS = 1

    best_valid_loss = float('inf')

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        # for epoch in trange(EPOCHS, desc="Epochs", disable=False):

        start_time = time.monotonic()

        # train_loss, train_acc = train_one_step(model, dataset['train'], dataloader['train'], optimizer, criterion, device)
        # valid_loss, valid_acc = evaluate(model, dataset['val'], dataloader['val'], criterion, device)
        train_loss, train_acc = train_one_step(model, dataloader['train'], dataset_size_train, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, dataloader['validation'], dataset_size_validation, criterion, device)

        log_train_accuracy.append(train_acc)
        log_train_loss.append(train_loss)
        log_valid_accuracy.append(valid_acc)
        log_valid_loss.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), '%s.pt' % model.model_name)
            if not os.path.exists(model_dst):
                os.makedirs(model_dst)
            save_model(model, optimizer, criterion, os.path.join(model_dst, '%s_tmp.pt' % model.model_name), epoch)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} \t | Train Acc: {train_acc * 100:.2f}%')
        print(f'\tVal. Loss: {valid_loss:.3f} \t | Val. Acc: {valid_acc * 100:.2f}%')
        print('='*20)

    if not os.path.exists(model_dst):
        os.makedirs(model_dst)
    save_model(model, optimizer, criterion, os.path.join(model_dst, '%s_final.pt' % model.model_name), epoch_num)

    ta_file = os.path.join(model_dst, 'total_train_accuracy_%s.txt' % (model.model_name))
    tl_file = os.path.join(model_dst, 'total_train_loss_%s.txt' % (model.model_name))
    va_file = os.path.join(model_dst, 'total_valid_accuracy_%s.txt' % (model.model_name))
    vl_file = os.path.join(model_dst, 'total_valid_loss_%s.txt' % (model.model_name))

    with open(ta_file, 'w') as total_accuracy_file:
        total_accuracy_file.write('\n'.join(['{:.4f}'.format(x) for x in log_train_accuracy]))
        # total_accuracy_file.write('\n'.join(total_accuracy))

    with open(tl_file, 'w') as total_loss_file:
        total_loss_file.write('\n'.join(['{:.4f}'.format(x) for x in log_train_loss]))
        # total_loss_file.write('\n'.join(total_loss))

    with open(va_file, 'w') as total_accuracy_file:
        total_accuracy_file.write('\n'.join(['{:.4f}'.format(x) for x in log_valid_accuracy]))
        # total_accuracy_file.write('\n'.join(total_accuracy))

    with open(vl_file, 'w') as total_loss_file:
        total_loss_file.write('\n'.join(['{:.4f}'.format(x) for x in log_valid_loss]))
        # total_loss_file.write('\n'.join(total_loss))

    # test_the_model(model, dataset['test'], dataloader['test'], criterion, device)
    pass


def train_one_step(model, dataloader, sample_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    running_acc = 0.0
    running_loss = 0.0

    mini_batches = 50

    # for (x, y) in tqdm(dataloader, desc="Training", leave=False):
    for i, (x, y) in enumerate(dataloader, 0):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        _, y_pred = torch.max(outputs, 1)

        loss = criterion(outputs, y)
        running_acc += torch.sum(y_pred == y.data)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        epoch_acc += torch.sum(y_pred == y.data)
        epoch_loss += loss.item()
        if i % mini_batches == mini_batches - 1:  # print every 2000 mini-batches
            print(f'[Batch: {i + 1:5d}] \t Accuracy: {running_acc / (y.shape[0] * mini_batches):.6f} \t Loss: {running_loss / (y.shape[0] * mini_batches):.6f}')
            running_acc = 0.0
            running_loss = 0.0

    return epoch_loss / sample_size, epoch_acc.float() / sample_size


def evaluate(model, iterator, sample_size, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    # sample_size = len(dataset)
    # model.eval()
    with torch.no_grad():
        # for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
        for i, (x, y) in enumerate(iterator, 0):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, y_pred = torch.max(outputs, 1)

            loss = criterion(outputs, y)
            correct = torch.sum(y_pred == y.data)

            epoch_loss += loss.item()
            epoch_acc += correct.item()
    return epoch_loss / sample_size, epoch_acc / sample_size


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def test_the_model(model, dataset, iterator, criterion=None, model_dst=model_dst):
    dataset_size_test = len(dataset)
    model_file_url = os.path.join(model_dst, '%s_final.pt' % model.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =================
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        model = model.to(device)
    # =================

    loaded_file = torch.load(model_file_url)
    model.load_state_dict(loaded_file['model_state_dict'])

    # model.load_state_dict(torch.load('tut4-model.pt'))
    # test_loss, test_acc = evaluate(model, dataset, iterator, criterion, device)
    test_loss, test_acc = evaluate(model, iterator, dataset_size_test, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


def get_predictions(model, iterator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in tqdm(iterator):
            x = x.to(device)
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def save_model(model, optimizer, criterion, model_file, epochs=0):
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'epochs': epochs
    }, model_file)
    print('Model saved', model_file)


def test_model_separate_accuracy(model, dataset, dataloader, criterion=None, model_dst=model_dst):
    mini_batches = 50
    classes_idx = dataset.classes
    emotions = [
        'Neutral',
        'Happiness',
        'Sadness',
        'Surprise',
        'Fear',
        'Disgust',
        'Anger',
        'Contempt',
        'None',
        'Uncertain',
        'No-Face',
    ]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Using device: {}'.format(device))

    # for camera_position in camera_positions:
    model_file_url = os.path.join(model_dst, '%s_final.pt' % model.model_name)
    loaded_file = torch.load(model_file_url)
    model.load_state_dict(loaded_file['model_state_dict'])
    model.to(device)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in emotions}
    total_pred = {classname: 0 for classname in emotions}

    # again no gradients needed
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            # correct_pred - в числитель - количество правельно распознанных эмоций
            # total_pred - в знаменатель - общее количество заданной эмоции
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[emotions[label]] += 1
                total_pred[emotions[label]] += 1

            if i % mini_batches == mini_batches - 1:  # print every 2000 mini-batches
                print(f'[Batch: {i + 1:5d}] \t {correct_pred}')

    # print accuracy for each class
    print('='*20, 'All positions', '='*20)
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname:5s} not available')

    s1_file = os.path.join(model_dst, 'emotion_correct_%s.txt' % (model.model_name))
    with open(s1_file, 'w') as file:
        file.write('\n'.join([f'{k}:\t{v}' for k, v in correct_pred.items()]))

    s2_file = os.path.join(model_dst, 'emotion_count_%s.txt' % (model.model_name))
    with open(s2_file, 'w') as file:
        file.write('\n'.join([f'{k}\t{v}' for k, v in total_pred.items()]))


def model_coatnet():
    # Train and validate model
    v_dataset, v_dataloader = ds_train_validation_all()
    v_classes = v_dataset['train'].dataset.classes
    v_model = coatnet_0(num_classes=len(v_classes))
    print('Trained model name is:', v_model.model_name)
    training_the_model(v_model, v_dataset, v_dataloader, epoch_num=epoch_number, lr=learning_rate)

    # Test model
    v_dataset_test, v_dataloader_test = ds_test_cam('all')
    test_the_model(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)
    test_model_separate_accuracy(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)


def model_vgg():
    # Train and validate model
    v_dataset, v_dataloader = ds_train_validation_all()
    v_classes = v_dataset['train'].dataset.classes
    v_model = VGG(in_channels=3, num_classes=len(v_classes))
    print('Trained model name is:', v_model.model_name)
    training_the_model(v_model, v_dataset, v_dataloader, epoch_num=epoch_number, lr=learning_rate)

    # Test model
    v_dataset_test, v_dataloader_test = ds_test_cam('all')
    test_the_model(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)
    test_model_separate_accuracy(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)

    # v_dataset, v_dataloader, v_classes = data_processing_val(batch_size=batch_size)
    # v_model = VGG(in_channels=3, num_classes=len(v_classes))
    # print('Trained model name is:', v_model.model_name)
    # training_the_model(v_model, v_dataset, v_dataloader, v_classes, epoch_num=epoch_number, lr=learning_rate)
    # # test_the_model(v_model, v_dataset['test'], v_dataloader['test'], criterion=None, device='cuda:0', model_dst='./models_trained')
    # test_model_separate_accuracy(v_model, batch_size, '../train_class', model_dst='./models_trained')


if __name__ == '__main__':
    # Hyper parameters imported from parameters.py
    model_coatnet()
    model_vgg()
