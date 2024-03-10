""" 
Plot images with corresponding bounding boxes from YOLO annotation format.
"""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_images_with_boxes(image_folder, label_folder):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    for image_file in image_files[:15]:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))

        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes from YOLO annotations
        with open(label_path, 'r') as label_file:
            for line in label_file:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert YOLO format to bounding box coordinates
                x, y, w, h = int((x_center - width / 2) * image.shape[1]), int((y_center - height / 2) * image.shape[0]), int(width * image.shape[1]), int(height * image.shape[0])

                # Plot bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)

        # Display the image with bounding boxes
        plt.imshow(image)
        plt.title(f'Image: {image_file}')
        plt.axis('off')
        plt.savefig(f'../sandbox/{image_file}.jpg')
        plt.cla()


if __name__ == "__main__":
    
    # Example usage
    image_folder_path = "/data/blanka/DATASETS/VisDrone/train/images"
    label_folder_path = "/data/blanka/DATASETS/VisDrone/train/labels"
    plot_images_with_boxes(image_folder_path, label_folder_path)
