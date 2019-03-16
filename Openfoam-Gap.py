import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use CPU
from keras.models import load_model
import numpy as np
from skimage import io, transform
from prednet import PredNet
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
# No Rolling prediction
# [19.38579128 19.40041641 19.41840862 19.4406757  19.45810326 19.46453638
#  19.47457091 19.48880379 19.4990681  19.50395397]
# [38040.68247266 38233.30683887 38409.83994043 39009.74465332
#  38716.9615459  38429.93100391 38026.84361133 37259.66738477
#  35966.71241211 35408.83411816]
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
  for i in range(0, (int)(n_frame / 2)):
    checksum_result[i] += checksum(sample_true[i, :, :, :], sample_true[i+(int)(n_frame / 2), :, :, :])
    # mse_result[i] += mse(sample_true[i, :, :, :],sample_true[i+(int)(n_frame / 2), :, :, :])
    mse_result[i] += PSNR(sample_true[i, :, :, :], sample_true[i + (int)(n_frame / 2), :, :, :])


checksum_result /= test_size
mse_result /=test_size
print(mse_result)
print(checksum_result)
