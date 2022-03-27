from sqlAlchemy_db import session, HeadPositionTrain, HeadPositionTest, HeadPositionValidation


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


if __name__ == '__main__':
    # insert_filenames(directory=folder_dst_dic['train'])
    # insert_filenames(directory='../val_class')
    # insert_filenames(directory='../test_class')
    print(f'The number of rows in table {HeadPositionTrain.__tablename__} is {session.query(HeadPositionTrain).count()}')
    print(f'The number of rows in table {HeadPositionValidation.__tablename__} is {session.query(HeadPositionValidation).count()}')
    print(f'The number of rows in table {HeadPositionTest.__tablename__} is {session.query(HeadPositionTest).count()}')
    pass
