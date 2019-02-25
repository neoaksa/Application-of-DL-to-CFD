import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use CPU
from keras.models import load_model
import numpy as np
from skimage import io, transform
from prednet import PredNet
from keras import backend as K


def checksum(imageA, imageB):
  return np.sum(imageA) - np.sum(imageB)

def mse(imageA, imageB):
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1] * 3)

  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err
# ------------
# Validaton
# Step = 1
# Frame = 10
# stride = 10
# Rolling prediction
# -------------

WIDTH = 160
HEIGHT = 128

seq = load_model('openfoam.h5',custom_objects = {'PredNet': PredNet})

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
test_size = 100

checksum_result = np.zeros((int)(n_frame/2),dtype=np.float)
mse_result = np.zeros((int)(n_frame/2),dtype=np.float)

for _ in range(test_size):
  start_frame = np.random.randint(0, all_images.shape[0] - n_frame)
  sample_true = all_images[start_frame:start_frame + n_frame, :, :, :]
  sample_prev = sample_true[:(int)(n_frame / 2), :, :, :]

  for j in range((int)(n_frame / 2) + 1):
    new_pos = seq.predict(sample_prev[np.newaxis, 0 - (int)(n_frame / 2):, :, :, :])
    # new_pos = seq.predict(sample_prev[np.newaxis, :, :, :, :)
    new = new_pos[::, -1, ::, ::, ::]
    sample_prev = np.concatenate((sample_prev, new), axis=0)

  for i in range((int)(n_frame/2)):
      index = i + (int)(n_frame/2)
      checksum_result[i] += checksum(sample_true[index, :, :, :], sample_prev[index,:,:,:])
      mse_result[i] += mse(sample_true[index, :, :, :], sample_prev[index,:,:,:])

checksum_result /= test_size
mse_result /=test_size
print(checksum_result)
print(mse_result)

