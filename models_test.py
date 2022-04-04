import logging

from discrete_categories import camera_positions
from models_train import test_the_model, test_model_separate_accuracy
from parameters import model_dst
from db_split_dataset import ds_test_all, ds_test_cam

from models.VGG import VGG
from models.CoAtNet import coatnet_0
from models.ResNet import ResNet50


def test_coatnet():
    # Test model
    v_dataset_test, v_dataloader_test = ds_test_all()
    v_classes = v_dataset_test.classes
    v_model = coatnet_0(num_classes=len(v_classes))
    test_the_model(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)

    for camera in camera_positions:
        v_dataset_test, v_dataloader_test = ds_test_cam(camera)
        if v_dataset_test and v_dataloader_test:
            test_model_separate_accuracy(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst, camera=camera)


def test_resnet():
    # Test model
    v_dataset_test, v_dataloader_test = ds_test_all()
    logging.info(f'Number of test examples: \t {len(v_dataset_test)}')
    v_classes = v_dataset_test.classes
    v_model = ResNet50(num_classes=len(v_classes))
    test_the_model(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)
    test_model_separate_accuracy(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst)

    for camera in camera_positions:
        v_dataset_test, v_dataloader_test = ds_test_cam(camera)
        logging.info('='*100)
        logging.info(f'Number of test examples: \t {len(v_dataset_test)}')
        if v_dataset_test and v_dataloader_test:
            test_model_separate_accuracy(v_model, v_dataset_test, v_dataloader_test, criterion=None, model_dst=model_dst, camera=camera)


if __name__ == '__main__':
    test_resnet()
    # test_coatnet()
    pass
