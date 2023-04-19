import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
import imutils

def read_image(img):
  img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (532, 260))

  return img
  
def high_pass_filter(img):

  # Aplica a transformada rápida de Fourier na imagem e centraliza baixa frequência
  f = np.fft.fft2(img)
  fshift = np.fft.fftshift(f)
  
  # Aplica filtro passa faixa
  rows, cols = img.shape
  
  mask = np.zeros((rows, cols), np.uint8)
  r1, r2 = 10, 220
  c1, c2 = 10, 220
  mask[r1:r2, c1:c2] = 1

  # Aplicar a máscara na imagem transformada
  fshift = fshift * mask

  # Aplicação da transformada inversa para pegar o retorno da imagem
  f_ishift = np.fft.ifftshift(fshift)
  img_inverted = np.fft.ifft2(f_ishift)
  img_inverted = np.real(img_inverted)

  return img_inverted

def segmentation(img):
  # Aplica o método de Otsu para separar a região da mama do fundo

  img = img.astype("uint8")
  # Aplica uma equalização de histograma para melhorar o contraste da imagem
  # img = cv2.equalizeHist(img)

  # Aplica um filtro gaussiano para suavizar a imagem
  img = cv2.GaussianBlur(img, (37, 37), 0)
  _, thresholded = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY)

  return thresholded

def black_filter(img):
    mask = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

    # Aplicar a máscara na imagem original
    img_black_filter = cv2.bitwise_and(img, img, mask=mask)

    return img_black_filter

def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def contrast(img):
    ALPHA_OFFSET = 1.40

    # Calcula os valores mínimo e máximo dos pixels
    min_val, max_val, _, _ = cv2.minMaxLoc(img)

    # Aplica a transformação linear para aumentar o contraste
    alpha = 255 / (max_val - min_val) # quanto maior o valor de alpha, maior o contraste
    beta = -alpha * min_val
    img_contrast = cv2.convertScaleAbs(img, alpha=alpha*ALPHA_OFFSET, beta=beta)

    return img_contrast

def convert_images():
    image_list = [i for i in os.listdir("./dataset")  if i.endswith(".pgm")]

    for image in image_list:
        # Carrega a imagem PGM
        img = read_image("./dataset/" + image)
        image = image.split('.')[0]
        
        # Salva a imagem no formato JPEG
        cv2.imwrite("./dataset/jpeg/" + image + ".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def polygon_contours(img_proc, img_base):

    PERIMETER_PRECISION = 0.04
    MIN_AREA = 200
    VERTICES = 4
    # Encontra os contornos na imagem
    contours, hierarchy = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selected_contours = []

    height, width = img_proc.shape[:2]
    # Verifica se cada contorno é uma curva fechada
    for contour in contours:

        if cv2.contourArea(contour) < MIN_AREA:
            continue
        # Obtém o retângulo delimitador do contorno
        x, y, w, h = cv2.boundingRect(contour)
        
        # Verifica se o retângulo delimitador não coincide com as bordas da imagem
        if x > 0 and y > 0 and x + w < width and y + h < height:
            # Calcula o comprimento do contorno
            perimeter = cv2.arcLength(contour, True)
            
            # Aproxima o contorno por um polígono
            approx = cv2.approxPolyDP(contour, PERIMETER_PRECISION*perimeter, True)
            
            # Verifica se o polígono tem um número de vértices próximo de 4
            if len(approx) > VERTICES:
                selected_contours.append(contour)

    selected_contours = sorted(selected_contours, key=lambda c: cv2.contourArea(c), reverse=True) 
    
    if len(selected_contours) > 0:
        img_base = cv2.cvtColor(img_base, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_base, [selected_contours[0]], -1, (127, 0, 255), 3)

    return img_base

def images_processing():
    images_path = glob.glob("./dataset/jpeg/*.jpeg")

    for image_path in images_path:

        image_name = image_path.split('/')[-1]
        img_base = img = read_image(image_path)
        img = contrast(img)     # aumenta o contraste da imagem
        img = black_filter(img) # aplica um filtro pra tirar os pixels pretos
        img = high_pass_filter(img)
        img = segmentation(img)
        cv2.imwrite("./results/segmented/segmented_" + image_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img = polygon_contours(img, img_base)
        cv2.imwrite("./results/with_contours/contours" + image_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])        
        

if '__main__' == __name__:
    images_processing()
    

    