import numpy as np
from models import *
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from sklearn import utils
from skimage.transform import rescale, resize
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def image_path(img_type, img_number):
    '''
    image_type: 'images' #original image, 'annotations' #crop-weed label, 'mask' #vegetation segmentation
    image_number: the number on the image name
    '''
    image_name = img_type[:-1]
    if img_number < 10:
        path = '../input/dataset-master/dataset-master/'+str(img_type)+'/00'+str(img_number)+'_'+str(image_name)+'.png'
    else:
        path = '../input/dataset-master/dataset-master/'+str(img_type)+'/0'+str(img_number)+'_'+str(image_name)+'.png'
    return path

def label_generator(number):
    annotation = cv2.imread(image_path('annotations', number))
    height = annotation.shape[0]
    width = annotation.shape[1]
   # channel = annotation.shape[2]
    labels = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            if np.all(annotation[i,j,:] == np.array([0,255,0])):
                labels[i,j,0] = 1
            elif np.all(annotation[i,j,:] == np.array([0,0,255])):
                labels[i,j,1] = 1
            elif np.all(annotation[i,j,:] == np.array([0,0,0])):
                labels[i,j,2] = 1
    return labels
    
x_train = np.zeros((120, 161, 216, 3))
y_train = np.zeros((120, 161, 216, 3))
x_test = np.zeros((20, 161, 216, 3))
y_test = np.zeros((20, 161, 216, 3))

#plt.figure(figsize=(8,5))

for i in range(40):
    image = cv2.imread(image_path('images',i+1))
    image_rescaled = rescale(image, 1.0 / 6.0, anti_aliasing=True)
    label = label_generator(i+1)
    label_rescaled = rescale(label, 1.0 / 6.0, anti_aliasing=True)
    x_train[i,:,:,:] = image_rescaled
    y_train[i,:,:,:] = label_rescaled
    x_train[40+i,:,:,:] = np.fliplr(image_rescaled)
    y_train[40+i,:,:,:] = np.fliplr(label_rescaled)
    x_train[80+1,:,:,:] = np.flipud(image_rescaled)
    y_train[80+i,:,:,:] = np.flipud(label_rescaled)
#    plt.subplot(8,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(x_train[i,:,:,:])
    
for i in range(20):
    image = cv2.imread(image_path('images',i+41))
    image_rescaled = rescale(image, 1.0 / 6.0, anti_aliasing=True)
    label = label_generator(i+41)
    label_rescaled = rescale(label, 1.0 / 6.0, anti_aliasing=True)
    x_test[i,:,:,:] = image_rescaled
    y_test[i,:,:,:] = label_rescaled

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


full_model = unet_res(n_classes=3)
full_model.compile(loss = 'binary_crossentropy',
             optimizer = 'Adam',
             metrics = [f1, mean_iou])

callbacks_full = [
    EarlyStopping(patience=8, verbose=1),
    ReduceLROnPlateau(patience=5, verbose=1),
    ModelCheckpoint('model-full-cwfid.h5', verbose=1, save_best_only=True)
]

model_full_history = full_model.fit(x = x_train,
                         y = y_train,
                         batch_size = 5,
                         epochs = 30,
                         verbose = 1,
                         validation_split = 0.2,
                          callbacks = callbacks_full
                         )
                         
i = 15
plt.figure()
plt.subplot(2,3,1)
plt.imshow(pred_full[i,:,:,2])
plt.subplot(2,3,4)
plt.imshow(y_test[i,:,:,2])
plt.subplot(2,3,2)
plt.imshow(pred_full[i,:,:,1])
plt.subplot(2,3,5)
plt.imshow(y_test[i,:,:,1])
plt.subplot(2,3,3)
plt.imshow(pred_full[i,:,:,0])
plt.subplot(2,3,6)
plt.imshow(y_test[i,:,:,0])
