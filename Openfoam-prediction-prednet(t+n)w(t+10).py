from keras.models import load_model
import numpy as np
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from prednet import PredNet
from keras import backend as K
from keras.layers import Input
from keras.models import Model

# ------------
# Output result using n timestep prediction through PredNet(t+1)/PredNet(t+10) Model
# -------------
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

WIDTH = 160
HEIGHT = 128
nt = 10

# Since PredNet(t+1) module is trained to output error, we need to reconstruct it to output prediction, but using same weights.
# change the h5 files, we can change the model
seq = load_model('openfoam.h5',custom_objects =  {'PredNet': PredNet,'extrap_loss':extrap_loss})
# Create testing model (to output predictions)
layer_config = seq.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=seq.layers[1].get_weights(), **layer_config)
input_shape = list(seq.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

all_images = []
path = 'Sample/'
for image_path in os.listdir(path):
  if image_path.endswith(".jpg"):
    img = io.imread(path+image_path , as_grey=False)
    # img = img[54:222,108:320,:] #168,212
    img = img[120:445, 216:640, :]  # 168,212
    img = transform.resize(img,(HEIGHT,WIDTH,3))
    all_images.append(img)

K.image_data_format() == 'channels_last'
all_images = np.asarray(all_images,dtype=np.float)

# predict
# predict 15 frame based on the given 15 frames
n_frame = 20
start_frame = np.random.randint(0,all_images.shape[0]-n_frame)
sample_true = all_images[start_frame:start_frame + n_frame, :, :, :]
sample_prev = sample_true[:nt,:,:,:]

# Rolling prediction
for j in range(n_frame-nt):
    new_pos = test_model.predict(sample_prev[np.newaxis, 0-nt:, :, :, :])
    # new_pos = seq.predict(sample_prev[np.newaxis, :, :, :, :])
    new = new_pos[::, -1, ::, ::, ::]
    sample_prev = np.concatenate((sample_prev, new), axis=0)

for i in range(nt, n_frame):
    index = i
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    # if i >= (int)(n_frame/2):
    #     ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    # else:
    #     ax.text(1, 3, 'Initial trajectory', fontsize=20)
    ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    toplot = sample_prev[index, ::, ::, ::]
    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = sample_true[index, ::, ::, ::]

    plt.imshow(toplot)
    plt.savefig('result/%i_animate.png' % (index+1))
