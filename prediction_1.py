from keras.applications.resnet import preprocess_input, decode_predictions
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

model = keras.models.load_model('varyok.best.hd5')
image_size = (224, 224)
img_path = 'C:/Users/Ceren/Desktop/busı_3/val/tumor/benign (109).png'
img = image.load_img(img_path, target_size=image_size)
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # toplu eksen

predictions = (model.predict(img_array) > 0.5).astype("int32")
#score = predictions[0]
print(predictions[0][0])
if predictions[0][0] == 0:
    predict = 'tümör var'
else:
    predict = 'tümör yok'
print('Tahmin: ', predict)