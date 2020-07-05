import re #regex para testar padrões
import cv2 #usado para abrir as imagens e gerar retangulos dos contornos nas imagens
import pytesseract #usado para OCR
from pytesseract import Output # usado para obter um dicionario de dados sobre a imagem
import matplotlib.pyplot as plt # usado para exibir a imagem

img = cv2.imread('../../../Ativadade2/Steps/testeGIT/WordSegmentation-master/src/rgCorreto.jpeg') #lendo a imagem
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #imagem em escalas de cinza
thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #imagem binária
d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
#em d possui todas as informações sobre a imagem (['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])


numbers_test = '[0-9]' #para testar padroes que contenham numeros de 0 a 9

n_boxes = len(d['text']) #tamanho de blocos de texto
j = 0 #variavel de controle (apenas os três primeiros blocos de números são relevantes)
for i in range(n_boxes):
    if int(d['conf'][i]) > 60: #taxa de confiança, caso de mais que 60 (alta), deve-se desenhar o retangulo na imagem
        if re.match(numbers_test, d['text'][i]): #caso encontre blocos com números, contornar os retangulos das imagens
            j = j + 1 #incremento
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i]) #coordenadas do bloco de texto
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)#contorna um retangulo verde na imagem, conforme as coordenadas
            print(d['text'][i]) #escreve bloco de número
            if j == 3: break # após o terceiro bloco de texto encontrado, sai do laço

plt.imshow(img) #imagem a exibir
plt.show() #exibição de imagem
