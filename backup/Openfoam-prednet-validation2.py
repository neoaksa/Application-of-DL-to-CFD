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


# ------------
# Validaton
# Step = 1
# Frame = 10
# stride = 10
# Rolling prediction
# [ 10308.82901505   9526.21107104  34915.88720723  16579.73490708
#   61396.91894987  31781.46347743  84690.73586392  60162.32658035
#  106963.18453892 102282.4705937 ]
# [  28.87626998   28.80654993   92.54345165   87.63491487  295.47359197
#   309.48264591  977.49561906 1003.25724559 2363.85681818 2354.14866792]
# -------------
# [11093.10252956 10779.82581746 33808.16884491 17886.88320082
#  54421.65518864 33739.03846789 69972.71768205 60838.42843345
#  92609.11364021 97756.29011096]
#  PSNR
# [33.46796608 33.5483901  28.39486816 28.6642055  23.41098372 23.21260912
#  18.2223942  18.09502536 14.38657506 14.40250871]

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
      # mse_result[i] += mse(sample_true[index, :, :, :], sample_prev[index,:,:,:])
      mse_result[i] += PSNR(sample_true[index, :, :, :], sample_prev[index, :, :, :])

checksum_result /= test_size
mse_result /=test_size
print(checksum_result)
print(mse_result)

