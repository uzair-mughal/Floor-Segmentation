import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json
from efficientnet.tfkeras import EfficientNetB4
from segmentation_models.metrics import IOUScore, FScore
from segmentation_models.losses import DiceLoss, BinaryFocalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_model():
    # loading custom losses
    dice_loss = DiceLoss()
    focal_loss = BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # loading model architecture from json file
    with open('./model/floor_segmentation_model.json', 'r') as json_file:
        json_savedModel = json_file.read()
    model_j = tf.keras.models.model_from_json(json_savedModel, custom_objects={
                                              'dice_loss_plus_1binary_focal_loss': total_loss, 'iou_score': IOUScore(threshold=0.5), 'f1-score': FScore(threshold=0.5)})
    # loading model weights
    model_j.load_weights('./model/floor_segmentation_model.h5')

    return model_j


def floor_segmentation(model, image_path):
    # defining iamge size
    target_size = (640, 640, 3)

    # loading RGB image
    image = Image.open(image_path).convert('RGB')
    wid, hei = image.size
    image = image.resize((target_size[0], target_size[1]), Image.ANTIALIAS)
    image_arr = np.array(image)
    image_arr = image_arr/255

    # floor segmentation
    pr_mask = model.predict(np.array([image_arr]))[0]
    pr_mask = np.stack((pr_mask[:, :, 0],)*3, axis=-1)

    # saving mask of image
    plt.imsave(os.path.join(os.getcwd(), 'masks', os.path.split(
        image_path)[1].split('.')[0]+'.png'), pr_mask)
    mask = Image.open(os.path.join(os.getcwd(), 'masks', os.path.split(
        image_path)[1].split('.')[0]+'.png')).convert("RGBA")
    pixdata_mask = mask.load()

    # coverting almost gray pixels to white
    width, height = mask.size
    for y in range(height):
        for x in range(width):
            if pixdata_mask[x, y] >= (80, 80, 80, 255):
                pixdata_mask[x, y] = (255, 255, 255, 255)

    image = image.convert("RGBA")
    pixdata_image = image.load()

    # making white pixels transparent
    for y in range(height):
        for x in range(width):
            if pixdata_mask[x, y] == (255, 255, 255, 255):
                pixdata_image[x, y] = (255, 255, 255, 0)

    # saving segmentated transparent image
    image = image.resize((wid, hei), Image.ANTIALIAS)
    image.save(os.path.join(os.getcwd(), 'results', os.path.split(
        image_path)[1].split('.')[0]+'.png'), "PNG")


def main():
    image_path = sys.argv[1]

    if not os.path.exists(os.path.join(os.getcwd(), 'masks')):
        os.makedirs(os.path.join(os.getcwd(), 'masks'))

    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        os.makedirs(os.path.join(os.getcwd(), 'results'))

    model = get_model()
    floor_segmentation(model, image_path)


if __name__ == '__main__':
    main()
