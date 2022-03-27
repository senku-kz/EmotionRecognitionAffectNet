from sqlalchemy import create_engine, Integer, String, Column

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

engine = create_engine('sqlite:///models_trained/sqlite3.db')  # используя относительный путь

Base = declarative_base()


class HeadPosition(Base):
    __tablename__ = 'm_head_position'
    id = Column(Integer(), primary_key=True)
    dataset = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False)
    camera_label = Column(String(200), nullable=False)
    class_label = Column(String(200), nullable=False)


Base.metadata.create_all(engine)

session = sessionmaker(bind=engine)
session = Session(bind=engine)


def insert_one():
    o1 = HeadPosition(
        dataset='test',
        filename='qwe.jpg',
        camera_label='Front',
        class_label='Smile'
    )
    session.add(o1)

    # v_session.new
    session.commit()


def insert_many():
    o1 = HeadPosition(
        dataset='test',
        filename='qwe1.jpg',
        camera_label='Front',
        class_label='Smile'
    )
    o2 = HeadPosition(
        dataset='test',
        filename='qwe2.jpg',
        camera_label='Front',
        class_label='Smile'
    )
    session.add_all([o1, o2])
    # v_session.new
    session.commit()


def select_all():
    q = session.query(HeadPosition).all()
    print(q)


def update_one():
    i = session.query(HeadPosition).first()
    i.camera_label = 'Left'
    session.add(i)
    session.commit()

# insert_one()
# insert_many()
# select_all()
update_one()