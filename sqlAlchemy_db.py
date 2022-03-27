from sqlalchemy import create_engine, Integer, String, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from parameters import db_sql_file


"""
pip install SQLAlchemy
"""


engine = create_engine(f'sqlite:///{db_sql_file}')
Base = declarative_base()


class HeadPositionTrain(Base):
    __tablename__ = 'm_affectnet_train'
    # id = Column(Integer(), primary_key=True)
    dataset = Column(String(200), nullable=False)
    class_idx = Column(Integer(), nullable=False)
    class_label = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False, primary_key=True)
    camera_label = Column(String(200))

    def __str__(self):
        return f'[{self.dataset}; {self.class_label}; {self.filename}; {self.camera_label}]'

    def __repr__(self):
        return f'[{self.dataset}; {self.class_label}; {self.filename}; {self.camera_label}]'


class HeadPositionValidation(Base):
    __tablename__ = 'm_affectnet_validation'
    # id = Column(Integer(), primary_key=True)
    dataset = Column(String(200), nullable=False)
    class_idx = Column(Integer(), nullable=False)
    class_label = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False, primary_key=True)
    camera_label = Column(String(200))

    def __str__(self):
        return f'[{self.dataset}; {self.class_label}; {self.filename}; {self.camera_label}]'

    def __repr__(self):
        return f'[{self.dataset}; {self.class_label}; {self.filename}; {self.camera_label}]'


class HeadPositionTest(Base):
    __tablename__ = 'm_affectnet_test'
    # id = Column(Integer(), primary_key=True)
    dataset = Column(String(200), nullable=False)
    class_idx = Column(Integer(), nullable=False)
    class_label = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False, primary_key=True)
    camera_label = Column(String(200))


# Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

session = sessionmaker(bind=engine)
session = Session(bind=engine)
