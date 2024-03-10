

from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os

def visdrone2yolo(dir):
    
    """
    Source: https://github.com/ultralytics/yolov5/blob/ba63208025fb27df31f4f02265631f72bbbba6a5/data/VisDrone.yaml#L34
    """

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    # Make labels directory
    labels_dir = os.path.join(dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    pbar = tqdm(Path(os.path.join(dir, 'annotations')).glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open(os.path.join(dir, 'images', f.stem + '.jpg')).size
        lines = []

        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        # Fix file path for labels
        labels_path = os.path.join(labels_dir, f.stem + '.txt')
        with open(labels_path, 'w') as fl:
            fl.writelines(lines)  # write label.txt

            
            
if __name__ == "__main__":
    
    folder = "/data/blanka/DATASETS/VisDrone/test"
    
    visdrone2yolo(folder)
