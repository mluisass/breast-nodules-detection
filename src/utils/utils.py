import glob
import os
import cv2
from matplotlib import pyplot as plt


def convert_PGM_images_to_JPEG():
    image_list = [i for i in os.listdir(PATH["dataset"])  if i.endswith(".pgm")]

    for image in image_list:
        # Carrega a imagem PGM
        img = read_image(PATH["dataset"] + image)
        image = image.split('.')[0]
        
        # Salva a imagem no formato JPEG
        cv2.imwrite(PATH["dataset"] + image + ".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def write_image(path, image_name, image):
    cv2.imwrite(PATH[path] + image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def write_text(text, img):
    org = (5, img.shape[0] - 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .8
    color = (127, 0, 255)
    thickness = 1
    lineType = cv2.LINE_AA
    cv2.putText(img, text, org, font, fontScale, color, thickness, lineType)

    return img


def get_all_images_path():
    return glob.glob(PATH["dataset"] + "*.jpeg")


def read_image(img):
    IMG_SIZE = (532, 260)

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)

    return img


def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_image_name(image_path):
    return image_path.split('/')[-1]
        

PATH = {
    "dataset": "dataset/",
    "high_contrast": "results/high_contrast/high_contrast_",
    "black_filter": "results/black_filter/black_filter_",
    "fft": "results/fft/fft_",
    "band_pass_filter": "results/band_pass_filter/band_pass_filter_",
    "ifft": "results/ifft/ifft_",
    "gaussian_blur": "results/gaussian_blur/gaussian_blur_",
    "segmented": "results/segmented/segmented_",
    "final_result": "results/final_result/final_result_",
}