from keras.applications import vgg16, inception_v3, resnet50, mobilenet

# from keras.applications import resnet

#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')
 
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')
