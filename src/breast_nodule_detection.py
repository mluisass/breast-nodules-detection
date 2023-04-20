import numpy as np
import cv2
from utils.utils import get_all_images_path, get_image_name, read_image, write_image, write_text

  
def fft_filter(img, image_name):
    # Aplica a transformada rápida de Fourier na imagem e centraliza baixa frequência
    fshift = fft(img, image_name)
    
    # Aplica filtro passa faixa
    fshift = band_pass_filter(img, image_name, fshift)

    # Aplicação da transformada inversa para pegar o retorno da imagem
    img_filtered = ifft(image_name, fshift)

    return img_filtered


def ifft(image_name, fshift):
    f_ishift = np.fft.ifftshift(fshift)

    img = np.fft.ifft2(f_ishift)
    img = np.real(img)

    write_image("ifft", image_name, img)

    return img


def band_pass_filter(img, image_name, fshift):
    INITIAL_ROW, END_ROW, INITIAL_COLUMN, END_COLUMN = 10, 220, 10, 220  

    rows, cols = img.shape

    # Criar máscara passa-faixa
    mask = np.zeros((rows, cols), np.uint8)
    mask[INITIAL_ROW:END_ROW, INITIAL_COLUMN:END_COLUMN] = 1

    # Aplicar a máscara na imagem transformada
    fshift = fshift * mask

    fshift_norm = cv2.normalize(np.abs(fshift), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    write_image("band_pass_filter", image_name, fshift_norm)

    return fshift


def fft(img, image_name):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    fshift_norm = cv2.normalize(np.abs(fshift), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    write_image("fft", image_name, fshift_norm)

    return fshift


def segmentation(img, image_name):
    SEGMENTATION_THRESHOLD = 35
    GAUSSIAN_K_SIZE = (37, 37)

    img = img.astype("uint8")

    # Aplica um filtro gaussiano para suavizar a imagem
    img = cv2.GaussianBlur(img, GAUSSIAN_K_SIZE, 0)

    write_image("gaussian_blur", image_name, img)

    # Aplica a limiarização
    _, thresholded = cv2.threshold(img, SEGMENTATION_THRESHOLD, 255, cv2.THRESH_BINARY)

    write_image("segmented", image_name, thresholded)

    return thresholded


def black_filter(img, image_name):
    BLACK_THRESHOLD = 100

    # Aplica a limiarização para os tons de cinza escuros se tornarem pretos (acentuação das bordas)
    mask = cv2.threshold(img, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # Aplicar a máscara na imagem original
    img_black_filter = cv2.bitwise_and(img, img, mask=mask)

    write_image("black_filter", image_name, img_black_filter)

    return img_black_filter



def high_contrast(img, image_name):
    # Quanto maior o valor de alpha, maior é o contraste.
    ALPHA_OFFSET = 1.40

    # Calcula os valores mínimo e máximo dos pixels
    min_val, max_val, _, _ = cv2.minMaxLoc(img)

    # Aplica a transformação linear para aumentar o contraste. 
    alpha = 255 / (max_val - min_val)
    beta = -alpha * min_val
    img_contrast = cv2.convertScaleAbs(img, alpha=alpha*ALPHA_OFFSET, beta=beta)

    write_image("high_contrast", image_name, img_contrast)

    return img_contrast


def polygon_contours(img_proc, img_base, image_name):
    PERIMETER_PRECISION = 0.04
    MIN_AREA = 200
    VERTICES = 4

    # Encontra os contornos na imagem
    contours, _ = cv2.findContours(img_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    selected_contours = []

    height, width = img_proc.shape
    
    # Verifica se cada contorno é uma curva fechada
    for contour in contours:

        if cv2.contourArea(contour) < MIN_AREA:
            continue
        
        if not_on_the_border(height, width, contour):
            # Calcula o comprimento do contorno
            perimeter = cv2.arcLength(contour, True)
            
            # Aproxima o contorno por um polígono
            approx = cv2.approxPolyDP(contour, PERIMETER_PRECISION*perimeter, True)
            
            # Verifica se o polígono tem um número de vértices próximo de 4
            if len(approx) > VERTICES:
                selected_contours.append(contour)

    selected_contours = sorted(selected_contours, key=lambda c: cv2.contourArea(c), reverse=True) 
    
    img_base = try_to_write_contour(img_base, image_name, selected_contours)

    return img_base


def try_to_write_contour(img_base, image_name, selected_contours):
    PINK_COLOR = (127, 0, 255)

    if len(selected_contours) > 0:
        img_base = cv2.cvtColor(img_base, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_base, [selected_contours[0]], -1, PINK_COLOR, 3)
        write_text("With Nodule", img_base)
    else:
        write_text("No Nodule", img_base)

    write_image("final_result", image_name, img_base)

    return img_base


def not_on_the_border(height, width, contour):
    # Obtém o retângulo delimitador do contorno
    x, y, w, h = cv2.boundingRect(contour)

    # Verifica se o retângulo delimitador não coincide com as bordas da imagem
    return (x > 0 and y > 0 and x + w < width and y + h < height)


def images_processing():
    images_path = get_all_images_path()

    for image_path in images_path:
        
        image_name = get_image_name(image_path)

        img_base = img = read_image(image_path)

        img = high_contrast(img, image_name)     
        
        img = black_filter(img, image_name)

        img = fft_filter(img, image_name)

        img = segmentation(img, image_name)

        img = polygon_contours(img, img_base, image_name)


if '__main__' == __name__:
    images_processing()
    

    