from skimage.filters import threshold_local  #Usado para obter uma imagem binária
import matplotlib.pyplot as plt #Usado para exibir as imagens
import numpy #usado para converter a imagem em um array numpy
from PIL import Image as im #usado para abrir a imagem
from scipy.ndimage import interpolation as inter #usado para rotacionar a imagem

#Passos para obter uma imagem binária
imageOriginal = im.open('../../../../datasetsTESTE/542794_991322_bundle_archive/scan_doc_rotation/images/scan_011.png')
imageNumpy = numpy.array(imageOriginal)
imageAux = threshold_local(imageNumpy, block_size=31, offset=40)
imageThreshold_local = (imageNumpy > imageAux)

img = imageThreshold_local ##Imagem binária obtida

#Exibe a imagem original
fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figheight(10)
fig.set_figwidth(10)
ax1.title.set_text('Imagem sem tratamento')
ax1.imshow(imageNumpy, cmap='gray')

#Função que busca o
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = numpy.sum(data, axis=1) #projetando a imagem horizontalmente para obter o histogram
    score = numpy.sum((hist[1:] - hist[:-1]) ** 2) #pontuações para cada angulo
    return hist, score

#delta é um valor que define o intervalo de angulos que a imagem pode ser gireda
#limite é um valor limite que delta pode rotacionar (tanto negativamente como positivamente)
delta = 1
limit = 5
angles = numpy.arange(-limit, limit+1, delta) #angulos para percorrer no loop abaixo (de -5 até 5)
scores = []

#Percorre os angulos e obtem as pontuações
for angle in angles:
    hist, score = find_score(img, angle)
    scores.append(score)

best_score = max(scores) #retorna a maior pontuação
best_angle = angles[scores.index(best_score)] #retorna o maior angulo
print('Best angle: {}'.format(best_angle))

#Corrigindo a imagem
data = inter.rotate(img, best_angle, reshape=False, order=0) # rotaciona a imagem no melhor angulo passado como parâmetro

#Exibe imagem com tratamento
ax2.title.set_text('Imagem com tratamento')
ax2.imshow(data, cmap='gray')
plt.show()
