import os
import numpy as np
import matplotlib.pyplot as plt

from parameters import model_dst


def create_plot_2_1(model_dst, model_name='CoAtNet', type='final'):
    steps = ['train', 'valid']

    for step in steps:
        acc_file = 'total_%s_%s_accuracy_%s.txt' % (step, model_name, type)
        acc_file_url = os.path.join(model_dst, acc_file)

        with open(acc_file_url, 'r') as total_accuracy_file:
            total_accuracy = [float(line.strip()) for line in total_accuracy_file]

        loss_file = 'total_%s_%s_loss_%s.txt' % (step, model_name,type)
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
        plot_file = 'accuracu-loss_%s_%s.png' % (model_name, step)
        plot_file_url = os.path.join(model_dst, plot_file)
        plt.savefig(plot_file_url)


def create_plot_4_1(model_dst, model_name='CoAtNet', type='final'):
    steps = ['train', 'valid']
    files = {}

    for step in steps:
        files[step] = {}

        files[step]['accuracy_filename'] = 'total_%s_%s_accuracy_%s.txt' % (step, model_name, type)
        files[step]['accuracy_file_url'] = os.path.join(model_dst, files[step]['accuracy_filename'])
        with open(files[step]['accuracy_file_url'], 'r') as total_accuracy_file:
            files[step]['accuracy'] = [float(line.strip()) for line in total_accuracy_file]

        files[step]['loss_filename'] = 'total_%s_%s_loss_%s.txt' % (step, model_name, type)
        files[step]['loss_file_url'] = os.path.join(model_dst, files[step]['loss_filename'])
        with open(files[step]['loss_file_url'], 'r') as total_loss_file:
            files[step]['loss'] = [float(line.strip()) for line in total_loss_file]

    # Plot losses
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, len(files[steps[0]]['accuracy'])), files[steps[0]]['accuracy'], label="Train accuracy")
    plt.plot(np.arange(0, len(files[steps[0]]['loss'])), files[steps[0]]['loss'], label="Train loss")
    plt.plot(np.arange(0, len(files[steps[1]]['accuracy'])), files[steps[1]]['accuracy'], label="Validation accuracy")
    plt.plot(np.arange(0, len(files[steps[1]]['loss'])), files[steps[1]]['loss'], label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.legend()
    plt.title('Accuracy - Loss chart')
    # plt.show()
    plot_file = 'accuracu-loss_%s_%s.png' % (model_name, 'Train_Validation')
    plot_file_url = os.path.join(model_dst, plot_file)
    plt.savefig(plot_file_url)


if __name__ == '__main__':
    # create_plot_2_1(model_dst, model_name='CoAtNet')
    # create_plot_2_1(model_dst, model_name='VGG')
    # create_plot_2_1(model_dst, model_name='ResNet')
    create_plot_4_1(model_dst, model_name='CoAtNet')
    create_plot_4_1(model_dst, model_name='VGG')
    create_plot_4_1(model_dst, model_name='ResNet')
