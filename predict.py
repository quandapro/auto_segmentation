import os
import cv2
import numpy as np
import segmentation_models as sm
import json
import warnings
warnings.filterwarnings('ignore')

IMAGE_FOLDER = 'data/images/'
MASKS_OUTPUT_FOLDER = 'data/masks/'
JSON_OUTPUT_FOLDER = 'data/images/'
IMAGE_WITH_CONTOUR_OUTPUT = 'data/images_with_contours'
LABEL = 'nuclei'

IMG_SIZE = (256, 256, 3)
MODEL_NAME = 'resnet34'
MODEL_WEIGHT_PATH = 'data/resnet34.h5'

model = sm.Unet(MODEL_NAME, input_shape=IMG_SIZE, classes=1, activation='sigmoid', encoder_weights=None)
model.load_weights(MODEL_WEIGHT_PATH)

def imageReader(imagePath):
    image = cv2.imread(imagePath)
    return image

def predict(image):
    h, w = image.shape[0:2]
    image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]))
    image = image.astype('float32') / 255.
    mask = model.predict(np.asarray([image]))
    mask = np.array(mask > 0.5, dtype='uint8').reshape(IMG_SIZE[0], IMG_SIZE[1])
    mask = cv2.resize(mask, (w, h))*255
    return mask

def saveMask(mask, maskPath):
    cv2.imwrite(maskPath, mask)

def contoursToJson(contours, label, height, width, imagePath, outputFile):
    x = {
        "version": "4.2.10",
        "flags": {},
        "shapes": [],
        "imagePath": imagePath,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    for contour in contours:
        contour = contour.reshape((contour.shape[0], contour.shape[2]))
        shape = {
            'label': label,
            'points': contour.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        x['shapes'].append(shape)

    json.dump(x, open(outputFile, 'w'))

def drawContoursOnImage(contours, image, outputPath):
    cv2.drawContours(image, contours, -1, (0,0,255), 5)
    cv2.imwrite(outputPath, image)

def main():
    image_files = os.listdir(IMAGE_FOLDER)
    image_name = [x.split('.')[0] for x in image_files]
    for i in range(len(image_files)):
        if ".json" in image_files[i] or ".git" in image_files[i]:
            continue

        image_path = os.path.join(IMAGE_FOLDER, image_files[i])
        output_mask_path = os.path.join(MASKS_OUTPUT_FOLDER, image_name[i] + '.jpg')
        output_json_path = os.path.join(JSON_OUTPUT_FOLDER, image_name[i] + '.json')
        output_image_with_contours_path = os.path.join(IMAGE_WITH_CONTOUR_OUTPUT, image_name[i] + '.jpg')

        # Load image
        image = imageReader(image_path)

        # Auto draw mask on image
        mask = predict(image)

        # Save mask 
        saveMask(mask, output_mask_path)

        #Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Save contours as labelme annotation
        contoursToJson(contours, LABEL, mask.shape[0], mask.shape[1], image_files[i], output_json_path)

        # Save images with contours
        drawContoursOnImage(contours, image, output_image_with_contours_path)

if __name__ == '__main__':
    main()





