from PIL import Image, ImageDraw


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]



def draw_image(img, bboxes):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        bbox_new = [bbox[1], bbox[3], bbox[0], bbox[2]] # xmin, ymin, xmax, ymax
        draw.rectangle(bbox, outline="red", width=2)
        print("iwas here")
        img.save("example.jpg")
    img.show()


image_filename = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/JPEGImages/2009_003066.jpg"
label_filename = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/YOLOLabels/2009_003066.txt"
bboxes = []

img = Image.open(image_filename)

with open(label_filename, 'r', encoding='utf8') as f:
    for line in f:
        data = line.strip().split(' ')
        bbox = [float(x) for x in data[1:]]
        bboxes.append(yolo_to_xml_bbox(bbox, img.width, img.height))

draw_image(img, bboxes)