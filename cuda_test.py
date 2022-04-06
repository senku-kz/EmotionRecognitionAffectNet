from db_sqlAlchemy import HeadPositionTrain, HeadPositionValidation, session


"""
pip install tensorflow
pip install keras
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
"""
if __name__ == '__main__':
    # Step 1: Check Pytorch (optional)
    import torch
    print("Cuda version: ", torch.__version__)
    print("Cuda available: ", torch.cuda.is_available())
    print("Cuda device: ", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name())

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print('GPU memory:', t, r, a, f)

    # Step 2: Check Tensorflow
    # from tensorflow.python.client import device_lib
    # print('-'*60)
    # print(device_lib.list_local_devices())

    # Step 3: Check Keras (optional)
    # from keras import backend as K
    # print(K.tensorflow_backend._get_available_gpus())

    # Step 4: Test DB count
    print('Test SQLite file:')
    print(f'Row count in {HeadPositionTrain.__tablename__} is {session.query(HeadPositionTrain).count()}')
    print(f'Row count in {HeadPositionValidation.__tablename__} is {session.query(HeadPositionValidation).count()}')
