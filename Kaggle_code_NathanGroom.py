##Kaggle Competition Code##
#username = nmg1987
# display name = nmg1987
##FIRST PART is the code I used to build the model to train (on all our training photos) ##
## SECOND PART is the code I used to test and predict the emotions for the Kaggle test photos ## 
##Part 1:
#this is the same model I'm turning in as the 'best model' for the final project
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers import Activation
import pandas as pd
from keras.optimizers import *
import glob
from PIL import Image
import os
import time
import datetime
from sklearn.cross_validation import KFold,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import keras.callbacks as cb
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


K.set_image_dim_ordering('th')
image_height=150
image_width=150
##these are the 'old' images
image_dir_in=r"/Users/nathangroom/desktop/new_docker/expressions/image_out/*.jpg"
legend_file_in=r"/Users/nathangroom/desktop/new_docker/expressions/data/legend.csv"

image_names=glob.glob(image_dir_in)
file=pd.read_csv(legend_file_in)
n_images=len(image_names)
emotions=['neutral', 'anger', 'surprise', 'sadness', 'happiness', 'contempt','disgust','fear']
i_emotions=[0,1,2,3,4,5,6,7]
d_emotions1=dict(zip(emotions,i_emotions))
d_emotions2=dict(zip(i_emotions,emotions))
num_classes=len(i_emotions)


X_image=[]
y_image=[]

actual_number=0

for i in range(n_images):
    try:
        
        im=Image.open(image_names[i])
        im = im.convert('L')  
        sz=im.size
        if sz != (image_height,image_width):
            print('The Image :- ', image_names[i] , ' is of - ', sz , ' dimension, which is incorrect' )
            continue


        im_arr= list(im.getdata())
        im_matrix=np.array(im_arr).reshape(image_height,image_width)   ## im has 3 layers

        im_nm = os.path.basename(image_names[i])
        im_emotion=file[file['image'] == im_nm]['emotion'].values



        if len(im_emotion) == 0:
            continue
        
        
    except:
        continue

    
    X_image.append(im_matrix)
    y_image.append( d_emotions1[im_emotion[0]] )
    actual_number += 1
X_image=np.reshape(X_image,(actual_number, 1, image_height,image_width))
X_image=X_image.astype('float16')

print('Total Images with macthing emotion : ' , actual_number)


X_train, X_test, y_train, y_test = train_test_split(X_image,y_image,test_size=0.2,random_state=22)
#seed=32
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 8)
Y_test = np_utils.to_categorical(y_test, 8)
batch_size = 84
nb_epoch = 11
nb_filters = 16
input_shape = (1, image_height, image_width)
nb_classes=len(i_emotions)
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

estop=cb.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=6, verbose=1, mode='auto')
pbl_callback=cb.ProgbarLogger()
plot_loss_callback = cb.LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))


def emotion_model():
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=( 1, image_height,image_width), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.summary()
    return model



print 'Training model...'
model=emotion_model()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=24,
          #callbacks = [estop],
          verbose=1, validation_data=(X_test, Y_test))


print "Training duration : {0}".format(time.time() - start_time)
score = model.evaluate(X_test, Y_test, batch_size=32)

print "Network's test score [loss, accuracy]: {0}".format(score)

##now the model is trained ##

##Part 2


#these are the new images. These have already been normalized in terms of size and color
image_dir_in=r"/users/nathangroom/desktop/new_docker/facial_expressions-master/test/*.jpg"
image_names=glob.glob(image_dir_in)
n_images=len(image_names)
X_image2=[]
for i in range(n_images):
    im=Image.open(image_names[i])
    im = im.convert('L') 
    im_arr= list(im.getdata())
    im_matrix=np.array(im_arr).reshape(image_height,image_width)
    im_nm = os.path.basename(image_names[i])
    X_image2.append(im_matrix)
X_image2=np.reshape(X_image2,(len(X_image2), 1, image_height,image_width))
X_image2=X_image2.astype('float16')
X_image2 /=  255

classes = model.predict_classes(X_image2, batch_size=32) 
##these are my answers for the Kaggle competition...
for i in classes:
    if i == 4:
        print 'happiness'
    if i == 6:
        print 'disgust'
    if i == 0:
        print 'neutral'
    if i ==3:
        print 'sadness'
    if i == 1:
        print 'anger'
    if i == 7:
        print 'fear'
    if i == 2:
        print 'surprise'
proba = model.predict_proba(X_image2, batch_size=32)
