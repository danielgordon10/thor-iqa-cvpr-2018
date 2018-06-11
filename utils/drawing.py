import pdb
import cv2
import numpy as np
import constants

BORDER = 0
CV_FONT = cv2.FONT_HERSHEY_DUPLEX

# plots: array of numpy array images to plot. Can be of different sizes and dimensions as long as they are 2 or 3 dimensional.
# rows: int number of rows in subplot. If there are fewer images than rows, it will add empty space for the blanks.
#       if there are fewer rows than images, it will not draw the remaining images.
# cols: int number of columns in subplot. Similar to rows.
# outputWidth: int width in pixels of a single subplot output image.
# outputHeight: int height in pixels of a single subplot output image.
# border: int amount of border padding pixels between each image.
# titles: titles for each subplot to be rendered on top of images.
# fancy_text: if true, uses a fancier font than CV_FONT, but takes longer to render.
def subplot(plots, rows, cols, outputWidth, outputHeight, border=BORDER,
        titles=None, fancy_text=False):
    returnedImage = np.full((
        (outputHeight + 2 * border) * rows,
        (outputWidth + 2 * border) * cols,
        3), 191, dtype=np.uint8)
    if fancy_text:
        from PIL import Image, ImageDraw, ImageFont
        FANCY_FONT = ImageFont.truetype(
            '/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf', 20)
    for row in range(rows):
        for col in range(cols):
            if col + cols * row >= len(plots):
                return returnedImage
            im = plots[col + cols * row]
            if im is None:
                continue
            if im.dtype != np.uint8 or len(im.shape) < 3:
                im = im.astype(np.float32)
                im -= np.min(im)
                im *= 255 / max(np.max(im), 0.0001)
                im = 255 - im.astype(np.uint8)
            if len(im.shape) < 3:
                im = cv2.applyColorMap(
                        im, cv2.COLORMAP_JET)
            if im.shape != (outputHeight, outputWidth, 3):
                imWidth = im.shape[1] * outputHeight / im.shape[0]
                if imWidth > outputWidth:
                    imWidth = outputWidth
                    imHeight = im.shape[0] * outputWidth / im.shape[1]
                else:
                    imWidth = im.shape[1] * outputHeight / im.shape[0]
                    imHeight = outputHeight
                imWidth = int(imWidth)
                imHeight = int(imHeight)
                im = cv2.resize(
                        im, (imWidth, imHeight),
                        interpolation=cv2.INTER_NEAREST)
                if imWidth != outputWidth:
                    pad0 = int(np.floor((outputWidth - imWidth) * 1.0 / 2))
                    pad1 = int(np.ceil((outputWidth - imWidth) * 1.0 / 2))
                    im = np.lib.pad(
                            im, ((0, 0), (pad0, pad1), (0, 0)),
                            'constant', constant_values=0)
                elif imHeight != outputHeight:
                    pad0 = int(np.floor((outputHeight - imHeight) * 1.0 / 2))
                    pad1 = int(np.ceil((outputHeight - imHeight) * 1.0 / 2))
                    im = np.lib.pad(
                            im, ((pad0, pad1), (0, 0), (0, 0)),
                            'constant', constant_values=0)
            if (titles is not None and len(titles) > 1 and
                    len(titles) > col + cols * row and
                    len(titles[col + cols * row]) > 0):
                if fancy_text:
                    if im.dtype != np.uint8:
                        im = im.astype(np.uint8)
                    im = Image.fromarray(im)
                    draw = ImageDraw.Draw(im)
                    for x in range(9,12):
                        for y in range(9, 12):
                            draw.text((x, y), titles[col + cols * row], (0,0,0),
                                    font=FANCY_FONT)
                    draw.text((10, 10), titles[col + cols * row], (255,255,255),
                            font=FANCY_FONT)
                    im = np.array(im)
                else:
                    scale_factor = max(max(im.shape[0], im.shape[1]) * 1.0 / 300, 1)
                    try:
                        cv2.putText(im, titles[col + cols * row], (10, 30), CV_FONT, .5 * scale_factor, [0,0,0], 4)
                        cv2.putText(im, titles[col + cols * row], (10, 30), CV_FONT, .5 * scale_factor, [255,255,255], 1)
                    except TypeError:
                        im = cv2.putText(im.copy(), titles[col + cols * row], (10, 30), CV_FONT, .5 * scale_factor, [0,0,0], 4)
                        im = cv2.putText(im.copy(), titles[col + cols * row], (10, 30), CV_FONT, .5 * scale_factor, [255,255,255], 1)
            returnedImage[
                border + (outputHeight + border) * row :
                        (outputHeight + border) * (row + 1),
                border + (outputWidth + border) * col :
                        (outputWidth + border) * (col + 1),:] = im
    im = returnedImage
    # for one long title
    if titles is not None and len(titles) == 1:
        if fancy_text:
            if im.dtype != np.uint8:
                im = im.astype(np.uint8)
            im = Image.fromarray(im)
            draw = ImageDraw.Draw(im)
            for x in range(9,12):
                for y in range(9, 12):
                    draw.text((x, y), titles[0], (0,0,0),
                            font=FANCY_FONT)
            draw.text((10, 10), titles[0], (255,255,255),
                    font=FANCY_FONT)
            im = np.array(im)
        else:
            scale_factor = max(max(im.shape[0], im.shape[1]) * 1.0 / 300, 1)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, .5 * scale_factor, [0,0,0], 4)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, .5 * scale_factor, [255,255,255], 1)

    return im

# BBoxes are [x1 y1 x2 y2]
def drawRect(image, bbox, padding, color):
    from utils import bb_util
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    bbox = np.round(np.array(bbox)) # mostly just for copying
    bbox = bb_util.clip_bbox(bbox, padding, imageWidth - padding, imageHeight - padding).astype(int).squeeze()
    padding = int(padding)
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[0]+padding+1] = color
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[2]-padding:bbox[2]+padding+1] = color
    image[bbox[1]-padding:bbox[1]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    image[bbox[3]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    return image

def draw_detection_box(image, box, cls, confidence=-1, color=None, width=1):
    #scale_factor = max(max(image.shape[0], image.shape[1]) * 1.0 / 300, 1)
    scale_factor = 1
    if color is None:
        import hashlib
        hex_digest = int(hashlib.sha1(cls.encode('utf-8')).hexdigest(), 16)
        r = int(hex_digest % 255)
        g = int((hex_digest / 255) % 255)
        b = int((hex_digest / 255**2) % 255)
        color = (b, g, r)
    color = tuple(map(int, color))
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, int(width * scale_factor))
    if confidence == 1 or confidence < 0:
        text_str = cls
    else:
        text_str = '%s: %d' % (cls, int(confidence * 100))
    text_size, baseline = cv2.getTextSize(text_str, CV_FONT, .5 * scale_factor, 1)
    text_size = list(text_size)
    #text_size[1] += baseline
    box[1] -= baseline
    if box[0] + text_size[0] > image.shape[1]:
        box[0] = image.shape[1] - text_size[0] - 1
    if box[1] - text_size[1] < 0:
        box[1] = text_size[1]
    text_coords = (int(box[0]), int(box[1]))
    cv2.rectangle(image, (text_coords[0], text_coords[1] + baseline),
            (text_coords[0] + text_size[0], text_coords[1] - text_size[1]),
            color, -1)
    cv2.putText(image, text_str, text_coords, CV_FONT, .5 * scale_factor, (0,0,0), 1)

def visualize_detections(image, boxes, classes, scores):
    out_image = image.copy()
    if len(boxes) > 0:
        try:
            boxes = (boxes / np.array([constants.SCREEN_HEIGHT * 1.0 / image.shape[1],
                    constants.SCREEN_WIDTH * 1.0 / image.shape[0]])[[0,1,0,1]]).astype(np.int32)
        except:
            pdb.set_trace()
            print('bad')
    for ii,box in enumerate(boxes):
        draw_detection_box(out_image, box, classes[ii], confidence=scores[ii], width=2)
    return out_image

