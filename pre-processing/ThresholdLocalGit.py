#Dataset utilizado: https://www.kaggle.com/c/denoising-dirty-documents/data
# As imagens desse data set, possui algumas marcas no fundo das imagens, lembrando marcas d'agua
# o threshold melhora a qualidade das imagens, "removendo" as marcas do fundo

import glob #Usado para obter os diretórios de todas as imagens
import numpy #usado para transformar a foto em uma matriz com 2 dimensões
from skimage.filters import threshold_local #Usado para aplicar o threshold local
from PIL import Image #Usado para abrir imagens"
from matplotlib import pyplot as plt #Usado para salvar imagens

#Array de nomes dos arquivos.
imagesPath = glob.glob('../../../datasetsTESTE/denoising-dirty-documents/train/*.png')

#Carrega todas as imagens
def loadImages ():
    imagesOpen = []
    images = []
    for i, value in enumerate(imagesPath):
        imagesOpen.append(Image.open(value))
        images.append(numpy.array(imagesOpen[i])) #indexando as imagens em matriz numpy no array images.
    return  images

images = loadImages() #todas as imagens convertidas em numpy array

#RemoveRuidos
def denoising():

    listImagesLocal = []
    listImagesBinary = []

    for i in range (143):
        listImagesLocal.append(threshold_local(images[i], block_size=35, offset=40))

        listImagesBinary.append(images[i] > listImagesLocal[i])

    return listImagesBinary

imagesTrated = denoising() #Lista das imagens binárias com thresholding local aplicado


#Salva as imagens com thsholding local aplicado
def saveImagesTrated():
    i = 0
    while i < len(imagesTrated):
        plt.imsave('/home/ryan/Documentos/LSCAD/Ativadade2/Steps/pre-processing/imagensBinarias_TLaplicado/{}.png'.format(i), imagesTrated[i], cmap = 'gray')
        i = i+1

a = saveImagesTrated()

