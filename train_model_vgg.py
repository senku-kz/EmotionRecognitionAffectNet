import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from models.VGG import VGG
from discrete_categories import camera_positions, cat_6


def imshow(imgs, title=None):
    # imshow for img(tensor)
    imgs = imgs.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imgs = std * imgs + mean
    imgs = np.clip(imgs, 0, 1)
    plt.imshow(imgs)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def check_loader(loader, class_names):
    # get some random training images
    data_iter = iter(loader)
    images, labels = data_iter.next()

    # print labels
    ttl = [class_names[x] for x in labels]
    print(ttl)
    # show images
    imshow(torchvision.utils.make_grid(images), ttl)
    pass


def gen_model_name(model_name, camera_position):
    camera_position = camera_position.replace(' ', '_')
    return '%s_%s.pth' % (model_name, camera_position)


def train_network(model, epochs_num, dataset, trainloader, criterion, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    model = model.to(device)

    total_loss = []
    total_accuracy = []

    set_size = len(dataset)

    for epoch in range(epochs_num):  # loop over the dataset multiple times
        print('Epoch {} / {}'.format(epoch+1, epochs_num))
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.6f}')

            # if i % 20 == 19:  # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
        epoch_loss = running_loss / set_size
        epoch_accuracy = running_corrects.float() / set_size
        print('Loss: {:.5f} Accuracy: {:.5f}'.format(epoch_loss, epoch_accuracy))

        total_loss.append(epoch_loss)
        total_accuracy.append(epoch_accuracy.item())

    print('Finished Training')
    return model, total_accuracy, total_loss


def create_model_and_save(data_dir, batch_size, epochs_number, model_dst, model_name='VGG16'):
    # Data Transformation and Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # step 5
    dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    class_names = dataset.classes

    # check_loader(dataloader, class_names)

    # Define a Convolutional Neural Network
    vgg16 = VGG(in_channels=3, num_classes=len(class_names))
    # netCoAtNet6 = CoAtNet6(num_classes=len(class_names))
    # netCoAtNet7 = CoAtNet7(num_classes=len(class_names))

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

    model, total_accuracy, total_loss = train_network(vgg16, epochs_number, dataset, dataloader, criterion, optimizer)

    if not os.path.exists(model_dst):
        os.makedirs(model_dst)
    camera_position = data_dir.split("/")[-1]
    camera_position = camera_position.replace(' ', '_')
    model_file = os.path.join(model_dst, '%s_%s.pth' % (model_name, camera_position))
    # torch.save(model.state_dict(), model_file)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, model_file)
    print('Model saved', model_file)

    ta_file = os.path.join(model_dst, 'total_accuracy_%s_%s.txt' % (model_name, camera_position))
    tl_file = os.path.join(model_dst, 'total_loss_%s_%s.txt' % (model_name, camera_position))

    with open(ta_file, 'w') as total_accuracy_file:
        total_accuracy_file.write('\n'.join(['{:.4f}'.format(x) for x in total_accuracy]))
        # total_accuracy_file.write('\n'.join(total_accuracy))

    with open(tl_file, 'w') as total_loss_file:
        total_loss_file.write('\n'.join(['{:.4f}'.format(x) for x in total_loss]))
        # total_loss_file.write('\n'.join(total_loss))

    pass


def create_plot(model_dst, camera_position, model_name='CoAtNet0'):
    v_camera = camera_position.replace(' ', '_')
    acc_file = 'total_accuracy_%s_%s.txt' % (model_name, v_camera)
    acc_file_url = os.path.join(model_dst, acc_file)

    with open(acc_file_url, 'r') as total_accuracy_file:
        total_accuracy = [float(line.strip()) for line in total_accuracy_file]

    loss_file = 'total_loss_%s_%s.txt' % (model_name, v_camera)
    loss_file_url = os.path.join(model_dst, loss_file)
    with open(loss_file_url, 'r') as total_loss_file:
        total_loss = [float(line.strip()) for line in total_loss_file]

    # Plot losses
    plt.figure(figsize=(10, 8))
    # plt.semilogy(np.arange(0, len(total_accuracy)), total_accuracy, label='accuracy')
    # plt.semilogy(np.arange(0, len(total_loss)), total_loss, label='loss')
    plt.plot(np.arange(0, len(total_accuracy)), total_accuracy, label="accuracy")
    plt.plot(np.arange(0, len(total_loss)), total_loss, label="loss")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    # plt.grid()
    plt.legend()
    plt.title('accuracy - loss chart')
    # plt.show()
    plot_file = 'accuracu-loss_%s_%s.png' % (model_name, camera)
    plot_file_url = os.path.join(model_dst, plot_file)
    plt.savefig(plot_file_url)


def test_model_common_accuracy(data_dir, model_dst='./models', camera_position='Forward', model_name='VGG16'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    testset = datasets.ImageFolder(data_dir, data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)
    class_names = testset.classes
    # model_dst = './models'

    # set_size = len(testloader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Using device: {}'.format(device))

    # check_loader(testloader, class_names)

    # Define a Convolutional Neural Network
    net = VGG(in_channels=3, num_classes=len(class_names))

    # for camera_position in camera_positions:
    model_file = gen_model_name(model_name=model_name, camera_position=camera_position)
    model_file_url = os.path.join(model_dst, model_file)

    loaded_file = torch.load(model_file_url)
    net.load_state_dict(loaded_file['model_state_dict'])
    net = net.to(device)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network {model_file} on the test images: {100 * correct // total} %')
    pass


def test_model_separate_accuracy(data_dir, model_dst='./models', camera_position='Forward', model_name='CoAtNet0'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    testset = datasets.ImageFolder(data_dir, data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)
    class_names = testset.classes
    # model_dst = './models'

    # set_size = len(testloader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Using device: {}'.format(device))

    # check_loader(testloader, class_names)

    # Define a Convolutional Neural Network
    net = VGG(in_channels=3, num_classes=len(class_names))

    # for camera_position in camera_positions:
    model_file = gen_model_name(model_name=model_name, camera_position=camera_position)
    model_file_url = os.path.join(model_dst, model_file)

    loaded_file = torch.load(model_file_url)
    net.load_state_dict(loaded_file['model_state_dict'])
    net = net.to(device)

    # ================
    classes = cat_6
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    print('='*20, camera_position, '='*20)
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname:5s} not available')


if __name__ == '__main__':
    print(torch.__version__)
    print('CUDA is available:', torch.cuda.is_available())
    labels = ['train', 'val', 'test']

    # for camera in camera_positions:
    #     batch_size = 8
    #     epochs_number = 1
    #     model_dst = './models'
    #
    #     data_dir = './data/train/%s' % camera
    #     test_dir = './data/test/%s' % camera
    #     # db_filename = './data/%s.sqlite3' % labels[0]
    #
    #     # create_model_and_save(data_dir, batch_size, epochs_number, model_dst, model_name='CoAtNet0')
    #
    #     test_model_common_accuracy(test_dir, model_dst, camera, model_name='CoAtNet0')
    #     test_model_separate_accuracy(test_dir, model_dst, camera, model_name='CoAtNet0')
    #
    #     create_plot(model_dst, camera, model_name='CoAtNet0')

    #     # test_network(test_dir)
    # net6 = CoAtNet6()
    # net7 = CoAtNet7()

    # batch_size = 8
    # camera_position = 'Forward'
    # test_dir = './data/test/%s' % camera_position
    # test_network(test_dir, camera_position, model_name='CoAtNet0')

    camera = 'All'
    batch_size = 4
    epochs_number = 50
    model_dst = './models_v2'

    data_dir = './data_v2/train/%s' % camera
    test_dir = './data_v2/test/%s' % camera
    # db_filename = './data/%s.sqlite3' % labels[0]

    # create_model_and_save(data_dir, batch_size, epochs_number, model_dst, model_name='VGG16')

    test_model_common_accuracy(test_dir, model_dst, camera, model_name='VGG16')
    test_model_separate_accuracy(test_dir, model_dst, camera, model_name='VGG16')

    create_plot(model_dst, camera, model_name='VGG16')
