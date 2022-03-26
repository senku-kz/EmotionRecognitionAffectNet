"""
pip install tensorflow
pip install keras
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
"""
if __name__ == '__main__':
    # Step 1: Check Pytorch (optional)
    import torch
    print("Cuda available: ", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name())

    # Step 2: Check Tensorflow
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # Step 3: Check Keras (optional)
    # from keras import backend as K
    # print(K.tensorflow_backend._get_available_gpus())
