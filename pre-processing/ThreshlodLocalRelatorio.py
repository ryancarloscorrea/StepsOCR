import numpy
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from PIL import Image

imageOriginal = Image.open('../../../datasetsTESTE/denoising-dirty-documents/train/2.png')
imageNumpy = numpy.array(imageOriginal)
imageAux = threshold_local(imageNumpy, block_size=35, offset=40)
imageThreshold_local = (imageNumpy > imageAux)

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(6)
fig.set_figwidth(5)
ax1.title.set_text('Imagem sem tratamento')
ax1.imshow(imageNumpy, cmap='gray')
ax2.title.set_text('Imagem com threshold local')
ax2.imshow(imageThreshold_local, cmap='gray')
plt.show()