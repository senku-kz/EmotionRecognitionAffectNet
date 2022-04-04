import os
import shutil
import numpy as np
from db_insert_records_into_sql import insert_filenames
from parameters import dataset_lst, folder_dst_lst


def sort_images_with_copy():
    dataset_lst = ['train_set', 'val_set']
    folder_dst_lst = ['../dataset/train', '../dataset/validation']
    cnt = {}
    for dataset, folder_dst in zip(dataset_lst, folder_dst_lst):
        folder_src = f'../data/{dataset}/images'
        folder_annotations = f'../data/{dataset}/annotations'
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
        cnt[dataset] = 0
        entries = os.listdir(folder_src)
        for entry in entries:
            if entry.endswith('.jpg'):
                filename = entry.split('.')[0]
                filename_annotation = os.path.join(folder_annotations, f'{filename}_exp.npy')
                if os.path.exists(filename_annotation):
                    # data_array = np.load(filename_annotation)
                    try:
                        data_array = np.load(filename_annotation, allow_pickle=True)
                        emotion_idx = data_array.item()
                        if not os.path.exists(os.path.join(folder_dst, emotions[emotion_idx])):
                            os.makedirs(os.path.join(folder_dst, emotions[emotion_idx]))
                        shutil.copyfile(
                            os.path.join(folder_src, entry),
                            os.path.join(folder_dst, emotions[emotion_idx], entry)
                        )
                        insert_filenames(dataset='validation', filename=entry, class_label=emotions[emotion_idx])
                        cnt[dataset] += 1
                    except:
                        print("Error file", filename_annotation)
        print(f'Copy {cnt[dataset]} files')
    print(f'Copy {cnt} files')


def insert_image_info_in_sql():
    cnt = {}
    for dataset, folder_dst in zip(dataset_lst, folder_dst_lst):
        folder_src = f'../data/{dataset}/images'
        folder_annotations = f'../data/{dataset}/annotations'
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
        cnt[dataset] = 0
        entries = os.listdir(folder_src)
        for entry in entries:
            if entry.endswith('.jpg'):
                filename = entry.split('.')[0]
                filename_annotation = os.path.join(folder_annotations, f'{filename}_exp.npy')
                if os.path.exists(filename_annotation):
                    # data_array = np.load(filename_annotation)
                    try:
                        data_array = np.load(filename_annotation, allow_pickle=True)
                        emotion_idx = data_array.item()
                        insert_filenames(dataset=folder_dst.split('/')[-1], filename=entry, class_idx=int(emotion_idx), class_label=emotions[emotion_idx])
                        cnt[dataset] += 1
                    except:
                        print("Error file", filename_annotation)
        print(f'Copy {cnt[dataset]} files')
    print(f'Copy {cnt} files')


if __name__ == '__main__':
    # sort_images_with_copy()
    insert_image_info_in_sql()
