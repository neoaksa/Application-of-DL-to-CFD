from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, transform

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
# translate sample into array
WIDTH = 42
HEIGHT = 53
n_frame = 30
filter = 24
sample_size = 100
epoch = 30
step_pred = 1  # step_pred < n_frame

seq = Sequential()
seq.add(ConvLSTM2D(filters=filter, kernel_size=(3, 3),
                   input_shape=(None, WIDTH, HEIGHT, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=filter, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=3, kernel_size=(3, 3, 3),
               activation='tanh',
               padding='same', data_format='channels_last'))
seq.compile(loss='mean_squared_error', optimizer='adadelta')

print(seq.summary())


all_images = []
path = 'Sample/'
for image_path in os.listdir(path):
  if image_path.endswith(".jpg"):
    img = io.imread(path+image_path , as_grey=False)
    img = img[54:222,108:320] #168,212
    img = transform.resize(img,(WIDTH,HEIGHT,3))
    # show image for testing
    # plt.imshow(img)
    # plt.show()
    all_images.append(img)

all_images = np.asarray(all_images,dtype=np.float)

# create training sample

x_all_samples = np.empty((0,n_frame,WIDTH,HEIGHT,3),dtype=np.float)
y_all_samples = np.empty((0,n_frame,WIDTH,HEIGHT,3),dtype=np.float)

for i in range(sample_size):
  start_frame = np.random.randint(0,all_images.shape[0]-n_frame-step_pred)
  x_sample = all_images[np.newaxis,start_frame:start_frame+n_frame,:,:,:]
  y_sample = all_images[np.newaxis,(start_frame+step_pred):(start_frame+n_frame+step_pred),:,:,:]
  x_all_samples = np.append(x_all_samples,x_sample,axis=0)
  y_all_samples = np.append(y_all_samples,y_sample,axis=0)

# train the model
seq.fit(x_all_samples, y_all_samples, batch_size=16,
        epochs=epoch, validation_split=0.05)
seq.save('openfoam.h5')