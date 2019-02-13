# Utilize prednet(coxlab) to predict CFD
import numpy as np
import os
from skimage import io, transform
from prednet import PredNet
from keras.layers import Input,Dense,Flatten
from keras.layers import TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import pylab as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0";

# Training parameters
nb_epoch = 100
batch_size = 10
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation
sample_size = 150
step_pred = 1
# Model parameters
K.image_data_format() == 'channels_last'
n_channels, im_height, im_width = (3, 128, 160)
input_shape = im_height, im_width, n_channels
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='prediction', return_sequences=True)
inputs = Input(shape=(nt,) + input_shape)
outputs = prednet(inputs)
# errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
# errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
# errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
# final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mean_squared_error', optimizer='RMSprop')
# print(seq.summary())

print(model.summary())

all_images = []
path = 'Sample/'
for image_path in os.listdir(path):
    if image_path.endswith(".jpg"):
        img = io.imread(path+image_path , as_grey=False)
        img = img[120:445,216:640,:] #168,212
        img = transform.resize(img,(im_height,im_width,3))
        # show image for testing
        # plt.imshow(img)
        # plt.show()
        all_images.append(img)

all_images = np.asarray(all_images,dtype=np.float)

# create training sample

x_all_samples = np.empty((0,nt,im_height,im_width,3),dtype=np.float)
y_all_samples = np.empty((0,nt,im_height,im_width,3),dtype=np.float)

for i in range(sample_size):
    start_frame = np.random.randint(0,all_images.shape[0]-nt-step_pred)
    x_sample = all_images[np.newaxis,start_frame:start_frame+nt,:,:,:]
    y_sample = all_images[np.newaxis,(start_frame+step_pred):(start_frame+nt+step_pred),:,:,:]
    x_all_samples = np.append(x_all_samples,x_sample,axis=0)
    y_all_samples = np.append(y_all_samples,y_sample,axis=0)

# train the model
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]

model.fit(x_all_samples, y_all_samples, batch_size=batch_size, callbacks=callbacks,
        epochs=nb_epoch, validation_split=0.05)
model.save('openfoam.h5')

