from keras.models import load_model
import numpy as np
import os
from skimage import io, transform
import matplotlib.pyplot as plt

WIDTH = 60
HEIGHT = 60
n_frame = 10
step_pred = 1
seq = load_model('openfoam.h5')

all_images = []
path = 'Sample/'
for image_path in os.listdir(path):
  if image_path.endswith(".jpg"):
    img = io.imread(path+image_path , as_grey=False)
    img = img[54:222,108:320,:] #168,212
    img = transform.resize(img,(WIDTH,HEIGHT,3))
    all_images.append(img)

all_images = np.asarray(all_images,dtype=np.float)

# predict
# predict 15 frame based on the given 15 frames

start_frame = np.random.randint(0,all_images.shape[0]-n_frame)
sample_true = all_images[start_frame:start_frame + n_frame + 10, :, :, :]
sample_prev = all_images[start_frame:start_frame + n_frame, :, :, :]

for j in range(10):
    new_pos = seq.predict(sample_prev[np.newaxis, -10:, ::, ::, ::])
    # new_pos = seq.predict(sample_prev[np.newaxis, :, :, :, :])
    print(new_pos.shape)
    new = new_pos[::, -1, ::, ::, ::]
    sample_prev = np.concatenate((sample_prev, new), axis=0)

for i in range(20):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    if i >= (int)(10):
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = sample_prev[i, ::, ::, ::]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = sample_true[i, ::, ::, ::]

    plt.imshow(toplot)
    plt.savefig('result/%i_animate.png' % (i + 1))