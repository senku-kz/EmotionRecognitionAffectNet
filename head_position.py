import os

from sqlalchemy import create_engine
from sqlalchemy import create_engine, Integer, String, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from HeadPoseEstimation import headPoseEstimation

from sqlAlchemy_db import session, HeadPositionTrain, HeadPositionTest, HeadPositionValidation

"""pip install SQLAlchemy"""


# engine = create_engine('sqlite:///models_trained/sqlite3.db')
# Base = declarative_base()
#
#
# class HeadPositionTrain(Base):
#     __tablename__ = 'm_head_position_train'
#     # id = Column(Integer(), primary_key=True)
#     dataset = Column(String(200), nullable=False)
#     class_label = Column(String(200), nullable=False)
#     filename = Column(String(200), nullable=False, primary_key=True)
#     camera_label = Column(String(200))
#
#
# class HeadPositionValidation(Base):
#     __tablename__ = 'm_head_position_validation'
#     # id = Column(Integer(), primary_key=True)
#     dataset = Column(String(200), nullable=False)
#     class_label = Column(String(200), nullable=False)
#     filename = Column(String(200), nullable=False, primary_key=True)
#     camera_label = Column(String(200))
#
#
# class HeadPositionTest(Base):
#     __tablename__ = 'm_head_position_test'
#     # id = Column(Integer(), primary_key=True)
#     dataset = Column(String(200), nullable=False)
#     class_label = Column(String(200), nullable=False)
#     filename = Column(String(200), nullable=False, primary_key=True)
#     camera_label = Column(String(200))
#
#
# # Base.metadata.drop_all(engine)
# Base.metadata.create_all(engine)
#
# session = sessionmaker(bind=engine)
# session = Session(bind=engine)


# def insert_filenames(directory='../test_class'):
#     for dirpath, dirnames, files in os.walk(directory):
#         for file_name in files:
#             if file_name.endswith('.jpg'):
#                 dataset_ttl = dirpath.split("\\")[-2].split('/')[-1]
#                 class_label = dirpath.split("\\")[-1]
#                 # file_name_uri = os.path.join(dirpath, file_name)
#                 print(dirpath, dataset_ttl, class_label, file_name)
#                 if dataset_ttl.startswith('train'):
#                     r = HeadPositionTrain(
#                         dataset=dataset_ttl,
#                         class_label=class_label,
#                         filename=file_name,
#                         # camera_label=headPoseEstimation(file_name_uri)
#                     )
#                 elif dataset_ttl.startswith('test'):
#                     r = HeadPositionTest(
#                         dataset=dataset_ttl,
#                         class_label=class_label,
#                         filename=file_name,
#                         # camera_label=headPoseEstimation(file_name_uri)
#                     )
#                 elif dataset_ttl.startswith('val'):
#                     r = HeadPositionValidation(
#                         dataset=dataset_ttl,
#                         class_label=class_label,
#                         filename=file_name,
#                         # camera_label=headPoseEstimation(file_name_uri)
#                     )
#                 else:
#                     continue
#
#                 session.merge(r)
#         session.commit()


def insert_filenames(dataset='train', filename='test.img', class_label='test'):
    print(dataset, filename, class_label)
    if dataset.startswith('train'):
        r = HeadPositionTrain(
            dataset=dataset,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )
    elif dataset.startswith('test'):
        r = HeadPositionTest(
            dataset=dataset,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )
    elif dataset.startswith('val'):
        r = HeadPositionValidation(
            dataset=dataset,
            class_label=class_label,
            filename=filename,
            # camera_label=headPoseEstimation(file_name_uri)
        )

    session.merge(r)
    session.commit()


def update_by_head_pose_estimation_train(directory='../'):
    q = session.query(HeadPositionTrain).filter(HeadPositionTrain.camera_label == None).all()
    for r in q:
        file_name_uri = os.path.join(directory, r.class_label, r.filename)
        r.camera_label = headPoseEstimation(file_name_uri)
        session.add(r)
        session.commit()
    pass


def update_by_head_pose_estimation_test(directory='../'):
    q = session.query(HeadPositionTest).filter(HeadPositionTest.camera_label == None).all()
    for r in q:
        file_name_uri = os.path.join(directory, r.class_label, r.filename)
        r.camera_label = headPoseEstimation(file_name_uri)
        session.add(r)
        session.commit()
    pass


if __name__ == '__main__':
    # insert_filenames(directory='../train_class')
    # insert_filenames(directory='../val_class')
    # insert_filenames(directory='../test_class')
    # print(f'The number of rows in table {HeadPositionTrain.__tablename__} is {session.query(HeadPositionTrain).count()}')
    # print(f'The number of rows in table {HeadPositionValidation.__tablename__} is {session.query(HeadPositionValidation).count()}')
    # print(f'The number of rows in table {HeadPositionTest.__tablename__} is {session.query(HeadPositionTest).count()}')

    # update_by_head_pose_estimation_train(directory='../data/train_set')
    # update_by_head_pose_estimation_test(directory='.\\data\\train_set\\images\\0.jpg')
    pass
