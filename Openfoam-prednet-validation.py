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
# Step = 11
# Frame = 11
# many to many
# [17865.09027735 12382.68024707 10720.88961328  8243.49399609
#   9147.70554199  8417.12629687  6969.72585058  5282.86579004
#   5347.4557959   6517.27092871]
# [77.1596937  46.73127345 36.2284855  30.48633629 25.61824698 22.42591293
#  20.13504016 19.28984422 18.89891779 19.70807476]

# -------------
# PSNR
# [22292.22177344 15941.17534961 13518.90217773 10334.18880566
#  10075.59331543  8554.57685547  7316.83952441  6605.83327344
#   6529.25114746  6685.31403809]
# [29.81042637 32.44932408 33.71259542 34.34375573 34.75052378 35.06318455
#  35.32546051 35.38265772 35.34990001 35.1510982 ]

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
n_frame = 11
step = 11
test_size = 100

checksum_result = np.zeros((n_frame-1),dtype=np.float)
mse_result = np.zeros((n_frame-1),dtype=np.float)

for _ in range(test_size):
  start_frame = np.random.randint(0,all_images.shape[0]-n_frame-step)
  sample_train = all_images[start_frame:start_frame + n_frame, :, :, :]
  sample_true = all_images[start_frame+step:start_frame + n_frame + step, :, :, :]
  newpos = seq.predict(sample_train[np.newaxis, :, :, :, :])

  for i in range(1,n_frame):
      checksum_result[i-1] += checksum(sample_true[i, :, :, :], newpos[0,i,:,:,:])
      # mse_result[i-1] += mse(sample_true[i, :, :, :], newpos[0,i,:,:,:])
      mse_result[i - 1] += PSNR(sample_true[i, :, :, :], newpos[0, i, :, :, :])

checksum_result /= test_size
mse_result /=test_size
print(checksum_result)
print(mse_result)

