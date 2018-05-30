import math
import numpy as np
import cv2

def extract_chip(image, bbox, scale_factor):
    '''
    Returns a subimage from provided 'image', cropped according to a rescaled 'bbox'

    :param image: (np array) 3 channel image as a numpy array which contains the face corresponding to 'bbox'
    :param bbox: (np array) 4 bounding box coordinates, in format [xmin, ymin, xmax, ymax]
    :param scale_factor: (float) Scalar to multiply box side, after it has been
    squared ie. put with width = height = largest bbox side
    :return: (tuple) A tuple containing the cropped image and the final bbox used ie. that was applied in 'image'
    '''

    # Square the bbox coordinates
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    side = np.max([width, height])

    if width > height:
        bbox[1] -= (side - height) / 2
        bbox[3] += (side - height) / 2
    else:
        bbox[0] -= (side - width) / 2
        bbox[2] += (side - width) / 2

    # Scale box
    final_box_side = side * scale_factor
    bbox[2] = bbox[0] + final_box_side
    bbox[3] = bbox[1] + final_box_side

    # Subtract offset to center after scaling
    offset = (final_box_side - side) / 2  # Divide by 2 to split offset between two sides
    bbox -= offset

    # 1. Extend image borders if bbox goes outside image boundaries

    # 1.1. Get image dims and initialize placeholders
    img_height, img_width, _channels = image.shape
    deltas_to_expand_img = np.zeros(4).astype('int16')  # Will hold how much we need to expand img on each side

    # 1.2. Check if any bbox dim goes outside borders and update borders
    if bbox[0] < 0:
        deltas_to_expand_img[0] = int(math.ceil(abs(bbox[0])))
    if bbox[1] < 0:
        deltas_to_expand_img[1] = int(math.ceil(abs(bbox[1])))
    if bbox[2] > img_width:
        deltas_to_expand_img[2] = int(math.ceil(bbox[2] - img_width))
    if bbox[3] > img_height:
        deltas_to_expand_img[3] = int(math.ceil(bbox[3] - img_height))

    # 1.3. Extend image with borders, replicating last border color
    #      Opencv function inputs are top, bottom, left, right, and we gathered
    #      borders in xmin ymin xmax ymax format

    extended_img = cv2.copyMakeBorder(image, deltas_to_expand_img[1], deltas_to_expand_img[3],
                                      deltas_to_expand_img[0], deltas_to_expand_img[2], cv2.BORDER_REPLICATE)

    # 2. Extract face chip ie. square face bounding box
    #    Since we padded the image and bbox coords are still in unpadded coord system we must add the offsets
    #    near the origin ie. deltas_to_expand_img[1] on y and deltas_to_expand_img[0] on x

    cropped_img = extended_img[
                  int(math.ceil(bbox[1] + deltas_to_expand_img[1])):
                  int(math.ceil(bbox[3] + deltas_to_expand_img[1])),
                  int(math.ceil(bbox[0] + deltas_to_expand_img[0])):
                  int(math.ceil(bbox[2] + deltas_to_expand_img[0]))
                  ]

    return (
        cropped_img,
        bbox
    )