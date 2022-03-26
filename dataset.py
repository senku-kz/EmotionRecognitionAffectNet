import os
import shutil
import numpy as np
from head_position import insert_filenames


def sort_train_images():
    folder_src = '../data/train_set/images'
    folder_annotations = '../data/train_set/annotations'
    folder_dst = '../dataset/train'
    emotions = {
        '0': 'Neutral',
        '1': 'Happiness',
        '2': 'Sadness',
        '3': 'Surprise',
        '4': 'Fear',
        '5': 'Disgust',
        '6': 'Anger',
        '7': 'Contempt',
        '8': 'None',
        '9': 'Uncertain',
        '10': 'No-Face',
    }
    if not os.path.exists(folder_src):
        print("No source folder")
        return False
    cnt = 0
    entries = os.listdir(folder_src)
    for entry in entries:
        if entry.endswith('.jpg'):
            filename = entry.split('.')[0]
            filename_annotation = os.path.join(folder_annotations, f'{filename}_exp.npy')
            if os.path.exists(filename_annotation):
                data_array = np.load(filename_annotation)
                emotion_idx = data_array.item()
                if not os.path.exists(os.path.join(folder_dst, emotions[emotion_idx])):
                    os.makedirs(os.path.join(folder_dst, emotions[emotion_idx]))
                shutil.copyfile(
                    os.path.join(folder_src, entry),
                    os.path.join(folder_dst, emotions[emotion_idx], entry)
                )
                insert_filenames(dataset='train', filename=entry, class_label=emotions[emotion_idx])
                cnt += 1
    print(f'Copy {cnt} files')


def sort_validation_images():
    dataset = 'val_set'
    folder_src = f'../data/{dataset}/images'
    folder_annotations = f'../data/{dataset}/annotations'
    folder_dst = '../dataset/validation'
    emotions = {
        '0': 'Neutral',
        '1': 'Happiness',
        '2': 'Sadness',
        '3': 'Surprise',
        '4': 'Fear',
        '5': 'Disgust',
        '6': 'Anger',
        '7': 'Contempt',
        '8': 'None',
        '9': 'Uncertain',
        '10': 'No-Face',
    }
    if not os.path.exists(folder_src):
        print("No source folder")
        return False
    cnt = 0
    entries = os.listdir(folder_src)
    for entry in entries:
        if entry.endswith('.jpg'):
            filename = entry.split('.')[0]
            filename_annotation = os.path.join(folder_annotations, f'{filename}_exp.npy')
            if os.path.exists(filename_annotation):
                data_array = np.load(filename_annotation)
                emotion_idx = data_array.item()
                if not os.path.exists(os.path.join(folder_dst, emotions[emotion_idx])):
                    os.makedirs(os.path.join(folder_dst, emotions[emotion_idx]))
                shutil.copyfile(
                    os.path.join(folder_src, entry),
                    os.path.join(folder_dst, emotions[emotion_idx], entry)
                )
                insert_filenames(dataset='validation', filename=entry, class_label=emotions[emotion_idx])
                cnt += 1
    print(f'Copy {cnt} files')


if __name__ == '__main__':
    sort_train_images()
    sort_validation_images()
