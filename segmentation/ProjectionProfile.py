#O código original pode ser encontrado em:
# https://github.com/ZeinabTaghavi/Handwriting_Manuscript_Line_and_Segment_Setection_Then_Storage/blob/master/line_detectoin/line_detection.py
#O código abaixo possui apenas algumas alterações, e foi adicionado vários comentários para melhor entendimento.
#- Ryan Carlos Coreêa
#
import cv2 #Usado para abrir imagem, converter para escalas de cinza, aplicar blur gaussiano,aplicar threshold, contornar blocos, entre outros.
import numpy as np # usado para manipular matrizes e vetores.
import matplotlib.pyplot as plt #usado para plotar gráficos e exibir imagens

def find_line_by_semi_histogram(img_file, vertical_percent, horizontal_percent, min_rect_size, min_border_percent):
    img = cv2.imread(img_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converte imagem para escala de cinza
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # aplicação do blur gaussian
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # ret é o limite (numérico), otsu é a imagem binária
    plt.title('Imagem binária')
    plt.imshow(otsu, cmap='gray')
    plt.show()
    gray_env = cv2.bitwise_not(otsu) #Onde é branco fica preto, onde é preto fica branco
    gray_corrected_rotation = otsu # otsu é a imagem após threshold
    plt.title('Imagem com as cores invertidas')

    plt.imshow(gray_env, cmap='gray')
    plt.show()
    # 2 - find the high compression vertical area

    vertical_hist = [sum(gray_env[i, :]) for i in range(img.shape[0])] #cada indice do vetor contem a somatorio de pixeis brancos de cada linha (letras)

    vertical_temp = gray_corrected_rotation.copy() #cópia da imagem gerada com a aplicação de threshold(otsu)
    plt.imshow(vertical_temp,cmap='gray')
    plt.show()
    vertical_limit = gray_env.shape[1] * 255 * vertical_percent * .01
    #histograma da imagem
    plt.title('Histograma vertical da imagem')
    n, bins, _ = plt.hist(vertical_hist, bins=20)
    plt.show()

    for i in range(len(vertical_hist)):
        if vertical_hist[i] > vertical_limit:
            vertical_temp[i, :] = 255 #toda linha da imagem fica somente branca caso
        else:
            vertical_temp[i, :] = 0 #toda linha da imgaem fica somente preta
    plt.title('Histograma vertical aplicado na imagem')
    plt.imshow(vertical_temp, cmap='gray')
    plt.show()

    contour, _ = cv2.findContours(vertical_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 3 - in vertical high compression areas, make all horizontal high compression areas

    vertical_lines_positions = []  # they are vertical high compression areas

    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt) # x, y coordenadas pixels, w largura, y altura do retangulo que cortorna a linha com mais ocorrencia de brancos
        print(cv2.boundingRect(cnt))
        vertical_lines_positions.append([y, y + h]) # os dois pontos formam uma linha vertical

    gray_corrected_rotation_env = cv2.bitwise_not(otsu) # imagem preto e branco (invertido)
    line_location_image = np.zeros((otsu.shape[0], otsu.shape[1]), np.uint8) #matriz do tamanho da imagem
    line_location_image.fill(255) #todos os pontos da matriz são brancos

    for y1, y2 in vertical_lines_positions:
        temp_img_env = gray_corrected_rotation_env[y1:y2, :] #imagem com tamanho do contorno com as cores invertidas
        horizontal_limit = (y2 - y1) * 255 * horizontal_percent * .01
        for j in range(temp_img_env.shape[1]):
            if sum(temp_img_env[:, j]) > horizontal_limit:
                line_location_image[y1:y2, j] = 0 #inicialmente esta variavel é uma matriz do tamanho da imagem toda branca, continua abaixo
                                                  #agora, toda vez que a ocorrencia de pixeis brancos(pretos anteriormente) for maior que o limite
                                                  #todos os pixeis do segmento da reta da coluna (y, y+h) serão pretos, formando assim uma imagem branca
                                                  #com varios retangulos pretos (onde ficam as letras)

    plt.title('Histograma horizontal aplicado na imagem')
    plt.imshow(line_location_image, cmap='gray')
    plt.show()
    kernel_h = int(img.shape[1] * .01)
    kernel_v = int(img.shape[0] * .004)

    print(kernel_v)
    dilate_kernel = np.ones((kernel_v, kernel_h), np.uint8)

    line_location_image = cv2.erode(line_location_image, dilate_kernel, iterations=1)
    plt.title('Erosão aplicada na imagem')
    plt.imshow(line_location_image, cmap='gray')
    plt.show()
    # cv2.imwrite(img_file + '_find_line_by_semi_histogram_2_just_lines.jpg', line_location_image)

    combine = cv2.bitwise_and(line_location_image, gray_corrected_rotation) #"sobreposição" da imagem original e a imagem branca com blocos pretos
                                                                            #esse passo é feito para adicionar os pixeis pretos que foram descartadados na projecao veritcal
    combine = cv2.dilate(combine, np.ones((8, 8), np.uint8), iterations=1)

    # cv2.imwrite(img_file + '_find_line_by_semi_histogram_3_horizontal_line_rected.jpg',combine)

    contour, _ = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # gray_corrected_rotation = cv2.erode(gray_corrected_rotation , kernel=dilate_kernel , iterations=0)
    cv2.drawContours(combine, contour, -1, 100, 10) #tirar essa linha depois
    plt.title('Imagem original combinada com a imagem com erosão')
    plt.imshow(combine, cmap='gray')
    plt.show()

    # cv2.imwrite(img_file + '_find_line_by_semi_histogram_4_contoured.jpg', combine)

    max_x = img.shape[1] * (1 - min_border_percent[1])
    min_x = img.shape[1] * min_border_percent[1]
    max_y = img.shape[0] * (1 - min_border_percent[0])
    min_y = img.shape[0] * min_border_percent[0]
    max_w = img.shape[1] * (1 - min_rect_size[1])
    min_w = img.shape[1] * min_rect_size[1]
    max_h = img.shape[0] * (1 - min_rect_size[0])
    min_h = img.shape[0] * min_rect_size[0]
    #
    # if not os.path.exists('lines_images_for_' + img_file):
    #     os.mkdir('lines_images_for_' + img_file)
    #
    count = 0
    #os.chdir('lines_images_for_' + img_file)
    for ctn in contour:
        (x, y, w, h) = cv2.boundingRect(ctn)
        if min_w < w < max_w and min_h < h < max_h and min_x < x < max_x and min_y < y < max_y:
            count += 1
            cv2.rectangle(otsu, (x, y - (int(h * 3 / 4))), (x + w, y + (int(h * 7 / 4))), (100, 100, 100), 4)
            line = img[y - (int(h * 3 / 4)): y + (int(h * 7 / 4)), x: x + w]

    plt.title('Os blocos contornados na imagem orignal')
    plt.imshow(otsu, cmap='gray')
    plt.show()

if __name__ == '__main__':

    n1 = 1
    n2 = 2

    avg_time = []

    for i in range(n1, n2):
        e1 = cv2.getTickCount()

        # per any type of documents set this things:
        erod_itter = 3
        dilate_itter = 1  # same for most of them
        ker_num = 2  # same for most of them
        th = 1
        bias = -150

        img_file = '/home/ryan/Documentos/LSCAD/Ativadade2/Steps/testeGIT/WordSegmentation-master/src/rgCorreto.jpeg'

        min_rect_size = [0.01, 0.01]  # percents of height and width of image
        min_border_percent = [0.05, 0.05]  # percents of image's height and width is border
        find_line_by_semi_histogram(img_file, 11, 10, min_rect_size, min_border_percent)
