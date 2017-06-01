import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img=mpimg.imread(fname="../data/train/2.png")
print(img.shape)
print(img.dtype)
plt.imshow(img)
plt.show()
