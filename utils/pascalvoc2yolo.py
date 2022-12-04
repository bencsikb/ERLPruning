""" Based on https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5 """
import xml.etree.ElementTree as ET
import glob
import os
import json

def xml_to_yolo_bbox_20078(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[0] + bbox[2]) / 2) / w
    y_center = ((bbox[1] + bbox[3]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def xml_to_yolo_bbox(bbox, w, h):
    # xmax, xmin, ymax, ymin
    x_center = ((bbox[0] + bbox[1]) / 2 ) / w    # xmax + xmin
    y_center = ((bbox[2] + bbox[3]) / 2 ) / h
    width = (bbox[0] - bbox[1]) / w
    height = (bbox[2] - bbox[3]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmax, xmin, ymax, ymin]


classes = []
input_dir = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/Annotations"
output_dir = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/YOLOLabels"
image_dir = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/JPEGImages"

# create the labels folder (output directory)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# identify all the xml files in the annotations folder (input directory)
files = glob.glob(os.path.join(input_dir, '*.xml'))
# loop through each
for fil in files:
    print(fil)
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
        print(f"{filename} image does not exist!")
        continue

    result = []

    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text
        # check for new classes and append to list
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [float(x.text) for x in obj.find("bndbox")]

        if '2007_' in fil or '2008_' in fil:
            yolo_bbox = xml_to_yolo_bbox_20078(pil_bbox, width, height)
        else:
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)

        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")

    if result:
        # generate a YOLO format text file for each xml file
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))

# generate the classes file as reference
with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))
