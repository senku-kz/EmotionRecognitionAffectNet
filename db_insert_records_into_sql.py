import os
import numpy as np
from db_sqlAlchemy import session, HeadPositionTrain, HeadPositionTest, HeadPositionValidation
from parameters import dataset_lst, folder_dst_lst


def insert_filenames(dataset='train', filename='test.img', class_idx=-1, class_label='test'):
    print('Insert to SQLite:', dataset, filename, class_idx, class_label)
    if dataset.startswith('train'):
        r = HeadPositionTrain(
            dataset=dataset,
            class_idx=class_idx,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )
    elif dataset.startswith('test'):
        r = HeadPositionTest(
            dataset=dataset,
            class_idx=class_idx,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )
    elif dataset.startswith('val'):
        r = HeadPositionValidation(
            dataset=dataset,
            class_idx=class_idx,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )

    session.merge(r)
    session.commit()


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
    # insert_filenames(directory=folder_dst_dic['train'])
    # insert_filenames(directory='../val_class')
    # insert_filenames(directory='../test_class')

    # print(f'The number of rows in table {HeadPositionTrain.__tablename__} is {session.query(HeadPositionTrain).count()}')
    # print(f'The number of rows in table {HeadPositionValidation.__tablename__} is {session.query(HeadPositionValidation).count()}')
    # print(f'The number of rows in table {HeadPositionTest.__tablename__} is {session.query(HeadPositionTest).count()}')
    insert_image_info_in_sql()
    pass
