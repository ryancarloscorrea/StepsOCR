import cv2 #Usado para ler a imagem, converter para escalas de cinza, aplicar threshold, desenhar retangulos
import pytesseract #usado para OCR
from pytesseract import Output #Usado para obter um dicionario como saida dos dados
import matplotlib.pyplot as plt #usado para exibir imagem

img = cv2.imread('../../../Ativadade2/Steps/testeGIT/WordSegmentation-master/src/rgCorreto.jpeg') #Lendo imagem
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#convertendo para escalas de cinza
thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #threshold aplicado

custom_config = r'--psm 6' #modo de segmentação por blocos de texto
d = pytesseract.image_to_data(thresh, 'por', custom_config,output_type=Output.DICT)
#em d possui todas as informações sobre a imagem (['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])

n_boxes = len(d['text']) #quantidade de blocos da imagem

for i in range(n_boxes):
    if int(d['conf'][i]) > 60: #taxa de confiança, caso de mais que 60 (alta), deve-se desenhar o retangulo na imagem
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i]) #coordenadas do bloco de texto
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # contorna um retangulo verde na imagem, conforme as coordenadas

plt.imshow(img) #imagem a exibir
plt.show() #exibe as imagens
