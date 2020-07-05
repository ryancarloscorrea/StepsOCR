import cv2 #usado para ler imagem, converter em escalas de cinza e aplicar threshold
import pytesseract #usado para reconhecer caracteres

img = cv2.imread('../../../Ativadade2/Steps/testeGIT/WordSegmentation-master/src/rgCorreto.jpeg') #lendo a imagem
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convertendo imagem para escalas de cinza
thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #imagem binária

custom_config = r'--oem 3 --psm 6' #customizações para melhor reconhecimento
print(pytesseract.image_to_string(thresh, 'por',config=custom_config)) #metodo que vai retornar o texto encontrado na imagem
