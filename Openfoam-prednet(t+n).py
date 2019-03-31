from keras.models import load_model
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use CPU
from skimage import io, transform
import matplotlib.pyplot as plt
from prednet import PredNet
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import pylab as plt
import time
from keras.callbacks import LearningRateScheduler
# ------------
# Build and train PredNet(t+n) model
# This model need trained PredNet(t+1) model 
# this model can do n timestep prediction
# nt - extrap_start_time is the rolling prediction steps
# -------------


# Define loss as MAE of frame predictions after t=0
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

nb_epoch=100
batch_size=10
im_width = 160
im_height = 128
sample_size = 150
nt = 20
extrap_start_time = 10 # in which timestep we start rolling prediction
# load module and configure
seq = load_model('openfoam.h5',custom_objects = {'PredNet': PredNet})

# Create testing model based on trained PredNet(t+1) model
layer_config = seq.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=seq.layers[1].get_weights(), **layer_config)
input_shape = list(seq.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(input_shape)
predictions = test_prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')

model.summary()

# pre-process samples
all_images = []
path = 'Sample/'
for image_path in sorted(os.listdir(path)):
    if image_path.endswith(".jpg"):
        img = io.imread(path+image_path , as_grey=False)
        img = img[120:445,216:640,:] #168,212
        img = transform.resize(img,(im_height,im_width,3))
        # # sqeeze color
        # sqeeze_color = np.vectorize(sqeeze_color, otypes=[np.float])
        # img[:, :, 0] = sqeeze_color(img[:, :, 0], SR * lowr, SR * highr)
        # img[:, :, 1] = sqeeze_color(img[:, :, 1], SG * lowg, SG * highg)
        # img[:, :, 2] = sqeeze_color(img[:, :, 2], SB * lowb, SB * highb)
        # # check zero
        # img_temp = img[:, :, 0] * img[:, :, 1] * img[:, :, 2]
        # for row in range(img_temp.shape[0]):
        #     for column in range(img_temp.shape[1]):
        #         if img_temp[row, column] == 0:
        #             for i in range(3):
        #                 img[row, column, i] = 0
        # show image for testing
        # plt.imshow(img)
        # plt.show()
        # print(image_path+" completed! \n")
        all_images.append(img)

all_images = np.asarray(all_images,dtype=np.float)
np.save('sample.npy',all_images)

# create training sample
x_all_samples = np.empty((0,nt,im_height,im_width,3),dtype=np.float)
y_all_samples = np.empty((0,nt,im_height,im_width,3),dtype=np.float)

for i in range(sample_size):
    start_frame = np.random.randint(0,all_images.shape[0]-nt)
    x_sample = all_images[np.newaxis,start_frame:start_frame+nt,:,:,:]
    x_all_samples = np.append(x_all_samples,x_sample,axis=0)


# train the model
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]


# evaluate performance of hardware
start_time = time.time()
history = model.fit(x_all_samples,x_all_samples,batch_size=batch_size, callbacks=callbacks,epochs=nb_epoch, validation_split=0.05)

print("--- %s seconds ---" % (time.time() - start_time))
model.save('openfoam_stepforward.h5')
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')
