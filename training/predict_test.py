import json
import numpy as np

import matplotlib.pyplot as plt


from os import path
from cv2 import resize
from resizeimage import resizeimage

model = load_model('model.h5')
train_datagen = ImageDataGenerator()
img_height, img_width = 120,160

img = resize(plt.imread(path.join('/home/jakub/test/1','img.jpg')), (img_width,img_height))
image = np.expand_dims(img, axis=0)
pred = model.predict(image, batch_size=1)

fig = plt.figure(figsize=(20, 20))

for i in range(1):
    fig.add_subplot(2,4,i+1).axis('Off')
    plt.imshow(img)
    x, y = pred[i]
    plt.scatter(x=[x], y=[y], c='r', s=5)
    print(x,y)

plt.show()

