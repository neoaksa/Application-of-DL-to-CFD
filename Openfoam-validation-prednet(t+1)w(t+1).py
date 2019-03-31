import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use CPU
from keras.models import load_model
import numpy as np
from skimage import io, transform
from prednet import PredNet
from keras import backend as K
import math
from keras.layers import Input
from keras.models import Model

def checksum(imageA, imageB):
  return abs(np.sum(imageA)*255 - np.sum(imageB)*255)

def mse(imageA, imageB):
  err = np.sum((imageA.astype("float")*255 - imageB.astype("float")*255) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1] * 3)

  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err

def PSNR(imageA, imageB):
  err = np.sum((imageA.astype("float") * 255 - imageB.astype("float") * 255) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1] * 3)
  if err == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(err))

def extrap_loss(y_true, y_hat):
  y_true = y_true[:, 1:]
  y_hat = y_hat[:, 1:]
  return 0.5 * K.mean(K.abs(y_true - y_hat),
                      axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)


# ------------
# validation the results from  1 timestep prediction through PredNet(t+1) Model
# -------------


WIDTH = 160
HEIGHT = 128
nt = 10
# Since PredNet(t+1) module is trained to output error, we need to reconstruct it to output prediction, but using same weights.
seq = load_model('openfoam_stepforward.h5',custom_objects = {'PredNet': PredNet,'extrap_loss':extrap_loss})
# Create testing model (to output predictions)
layer_config = seq.layers[1].get_config()
layer_config['output_mode'] = 'prediction' # config to prediction 
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=seq.layers[1].get_weights(), **layer_config)
input_shape = list(seq.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
validation_model = Model(inputs=inputs, outputs=predictions)

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
n_frame = 10
test_size = 100 # use 100 samples to validate

checksum_result = np.zeros((n_frame),dtype=np.float)
mse_result = np.zeros((n_frame),dtype=np.float)

for _ in range(test_size):
  start_frame = np.random.randint(0, all_images.shape[0] - nt - n_frame)
  sample_true = all_images[start_frame:start_frame + nt + n_frame, :, :, :]
  sample_prev = sample_true[:nt, :, :, :]
  for j in range(n_frame):
    new_pos = validation_model.predict(sample_prev[np.newaxis, 0 - nt:, :, :, :])
    # new_pos = seq.predict(sample_prev[np.newaxis, :, :, :, :)
    new = new_pos[::, -1, ::, ::, ::]
    sample_prev = np.concatenate((sample_prev, new), axis=0)

  for i in range(nt, nt + n_frame):
      checksum_result[i - nt] += checksum(sample_true[i, :, :, :], sample_prev[i,:,:,:])
      # mse_result[i-1] += mse(sample_true[i, :, :, :], newpos[0,i,:,:,:])
      mse_result[i - nt] += PSNR(sample_true[i, :, :, :], sample_prev[i, :, :, :])

checksum_result /= test_size
mse_result /=test_size
print(checksum_result)
print(mse_result)

