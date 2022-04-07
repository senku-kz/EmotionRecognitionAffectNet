from torchsummary import summary
from models.CoAtNet import coatnet_0
from models.ResNet import ResNet50
from models.VGG import VGG
from parameters import CUDA_N

print('>'*100)
model_coatnet = coatnet_0(num_classes=8)
model_coatnet.to(CUDA_N)
summary(model_coatnet, (3, 224, 224))

print('>'*100)
model_resnet = ResNet50(img_channel=3, num_classes=8)
model_resnet.to(CUDA_N)
summary(model_resnet, (3, 224, 224))

print('>'*100)
model_vgg = VGG(in_channels=3, num_classes=8)
model_vgg.to(CUDA_N)
summary(model_vgg, (3, 224, 224))
