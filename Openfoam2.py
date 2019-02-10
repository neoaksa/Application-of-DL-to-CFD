from keras.models import Input, Model
from keras.layers import MaxPooling2D, TimeDistributed, Conv2D, Conv3D
from keras.layers import UpSampling2D,add
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
import os
from skimage import io, transform

os.environ["CUDA_VISIBLE_DEVICES"]="1";
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
# translate sample into array
WIDTH = 60
HEIGHT = 60
n_frame = 30
filter = 32
sample_size = 100
epoch = 200
step_pred = 1  # step_pred < n_frame


input_img = Input(shape=(None,WIDTH, HEIGHT, 3))

x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(input_img)
x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
c1 = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
x = TimeDistributed(MaxPooling2D((2,2),(2,2)))(c1)

x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
c2 = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
x = TimeDistributed(MaxPooling2D((2,2),(2,2)))(c2)
x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
x = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
c3 = ConvLSTM2D(filters=filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)

x = TimeDistributed(UpSampling2D((2,2)))(c3)
x = add(([c2, x]))
x = TimeDistributed(Conv2D(filter,(3,3), padding='same'))(x)

x = TimeDistributed(UpSampling2D((2,2)))(x)
x = add(([c1, x]))
# x = TimeDistributed(Conv2D(filter,(3,3), padding='same'))(x)

print(x.shape)
x = TimeDistributed(Conv2D(filter,(3,3),padding='same'))(x)
x = TimeDistributed(UpSampling2D((2,2)))(x)
x = TimeDistributed(Conv2D(filter,(3,3),padding='same'))(x)
x = TimeDistributed(UpSampling2D((2,2)))(x)

x = TimeDistributed(MaxPooling2D((2,2),(2,2)))(x)
x = TimeDistributed(MaxPooling2D((2,2),(2,2)))(x)

output = Conv3D(filters=3, kernel_size=(3, 3, 3),
               activation='tanh',
               padding='same', data_format='channels_last')(x)
print(output.shape)

seq = Model(input_img, output=[output])

seq.compile(loss='mean_squared_error', optimizer='adadelta')

# print(seq.summary())

all_images = []
path = 'Sample/'
for image_path in os.listdir(path):
    if image_path.endswith(".jpg"):
        img = io.imread(path+image_path , as_grey=False)
        img = img[54:222,108:320,:] #168,212
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

