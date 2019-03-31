import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use CPU
from keras.models import load_model
import numpy as np
from skimage import io, transform
from prednet import PredNet
from keras import backend as K
import math


def checksum(imageA, imageB):
  return abs(np.sum(imageA)*255 - np.sum(imageB)*255)

# MSE
def mse(imageA, imageB):
  err = np.sum((imageA.astype("float")*255 - imageB.astype("float")*255) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1] * 3)
  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err

# Peak Signal-to-Noise Ratio
def PSNR(imageA, imageB):
  err = np.sum((imageA.astype("float") * 255 - imageB.astype("float") * 255) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1] * 3)
  if err == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(err))


# ------------
# Validaton
# Step = 1
# Frame = 10
# stride = 10
# No Rolling prediction
# [43985.30752051 11222.81349219 11153.26724805 17811.87954395
#   9831.43680859  9934.6330293  11297.64302344 12578.48803125
#  10088.30156543  8899.07063184]
# MSE
# [543.50200197  88.34318476  45.02813996  34.72857393  30.8319811
#   29.1567381   28.36796219  28.31829077  28.49429183  28.9494957 ]
# --------------
# [46052.8502627  12020.78884961 12434.20889063 19184.39918653
#   8835.7870625   8860.50340527 11166.58394824 13338.28942578
#  11780.0615     10569.36994141]
#  PSNR
# [20.78463416 28.7122242  31.63142456 32.77288427 33.32508228 33.60067562
#  33.70092881 33.68744208 33.63401844 33.58628664]
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
  new_pos = seq.predict(sample_prev[np.newaxis, :, :, :, :])


  for i in range(1,(int)(n_frame/2)+1):
      checksum_result[i-1] += checksum(sample_true[i, :, :, :], new_pos[0,i-1,:,:,:])
      # mse_result[i-1] += mse(sample_true[i, :, :, :], new_pos[0,i-1,:,:,:])
      mse_result[i - 1] += PSNR(sample_true[i, :, :, :], new_pos[0, i - 1, :, :, :])

checksum_result /= test_size
mse_result /=test_size
print(checksum_result)
print(mse_result)

