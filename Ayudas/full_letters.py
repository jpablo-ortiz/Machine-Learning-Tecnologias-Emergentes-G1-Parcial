import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from sklearn.model_selection import train_test_split
import numpy as np
import scipy


print(tf.__version__)
print(np.__version__)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# size of the image
pic_size = 48

# input path for the images
base_path = r"C:\Users\DELLPHOTO\Proyecto de Grado\Proyecto de grado\Datasets\BIG DATASET"
batch_size = 100
datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()
train_generator = datagen_train.flow_from_directory(
    base_path + r"\asl_alphabet_train", 
    target_size=(pic_size,pic_size), 
    color_mode="rgb", 
    batch_size=batch_size, \
    class_mode='categorical',shuffle=True)
validation_generator = datagen_validation.flow_from_directory(
    base_path + r"\asl_alphabet_test", 
    target_size=(pic_size,pic_size), 
    color_mode="rgb", 
    batch_size=batch_size, 
    class_mode='categorical', 
    shuffle=False)


model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
#model.trainable = False
num_classes=29
x = model.output

x = GlobalAveragePooling2D()(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=model.input, outputs=predictions)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#start training

epochs = 2

from keras.callbacks import ModelCheckpoint


checkpoint_filepath=r"C:\Users\DELLPHOTO\Proyecto de Grado\Proyecto de grado\callback/ALLmodel_weights7.h5"
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#callback

history = model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs=epochs, validation_data = validation_generator, validation_steps = validation_generator.n//validation_generator.batch_size, callbacks=callbacks_list)

# plot change in loss and accuracy
ruta_save=r"C:\Users\DELLPHOTO\Proyecto de Grado\Proyecto de grado\modeloCompleto"
#model.save(ruta_save) 

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss - categorical_crossentropy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
plt.savefig(r"C:\Users\DELLPHOTO\Proyecto de Grado\Proyecto de grado\graficas5")