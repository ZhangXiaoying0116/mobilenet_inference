# import keras
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf
import keras2onnx

model = MobileNet(include_top=True, weights='imagenet')
##convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'mobilenet_v1_keras-op13-fp32.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)

