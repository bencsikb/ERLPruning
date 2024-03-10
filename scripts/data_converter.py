import os
import random
import shutil
import numpy as np
import csv

def split_kitti(origpath, train_idxpath, val_idxpath):

    img_ext, label_ext = "image_3", "label_3"

    with open(train_idxpath) as f:
        reader = csv.reader(f)
        for line in reader:
            train_idx_list = line

    with open(val_idxpath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
           val_idx_list = row
        #val_idx_list = list(reader)

    print(len(train_idx_list), len(val_idx_list))


    if not os.path.isdir(os.path.join(origpath.replace("original", "training"), img_ext)):
        os.mkdir(os.path.join(origpath.replace("original", "training"), img_ext))
    if not os.path.isdir(os.path.join(origpath.replace("original", "training"), label_ext)):
        os.mkdir(os.path.join(origpath.replace("original", "training"), label_ext))


    split_folder(os.path.join(origpath, img_ext), os.path.join(origpath.replace("original", "training"), img_ext), train_idx_list, imgformat=".png")
    split_folder(os.path.join(origpath, label_ext), os.path.join(origpath.replace("original", "training"), label_ext), train_idx_list,  imgformat=".png")


    if not os.path.isdir(os.path.join(origpath.replace("original", "validation"), img_ext)):
        os.mkdir(os.path.join(origpath.replace("original", "validation"), img_ext))
    if not os.path.isdir(os.path.join(origpath.replace("original", "validation"), label_ext)):
        os.mkdir(os.path.join(origpath.replace("original", "validation"), label_ext))


    split_folder(os.path.join(origpath, img_ext), os.path.join(origpath.replace("original", "validation"), img_ext), val_idx_list,  imgformat=".png")
    split_folder(os.path.join(origpath, label_ext), os.path.join(origpath.replace("original", "validation"), label_ext), val_idx_list,  imgformat=".png")

def split_pascalvoc(origpath, train_idxpath, val_idxpath):

    img_ext, label_ext = "image", "label"

    with open(train_idxpath) as f:
        reader = csv.reader(f)
        for line in reader:
            train_idx_list = line

    with open(val_idxpath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
           val_idx_list = row
        #val_idx_list = list(reader)

    print(len(train_idx_list), len(val_idx_list))


    split_folder(os.path.join(origpath, "temp/images"), os.path.join(origpath, "validation/images"), train_idx_list, imgformat=".jpg")
    split_folder(os.path.join(origpath, "temp/labels"), os.path.join(origpath, "validation/labels"), train_idx_list, imgformat=".jpg")

    split_folder(os.path.join(origpath, "temp/images"), os.path.join(origpath, "test/images"), val_idx_list, imgformat=".jpg")
    split_folder(os.path.join(origpath, "temp/labels"), os.path.join(origpath, "test/labels"), val_idx_list, imgformat=".jpg")

    #split_folder(os.path.join(origpath, "JPEGImages"), os.path.join(origpath, "temp/images"), val_idx_list, imgformat=".jpg")
    #split_folder(os.path.join(origpath, "YOLOLabels"), os.path.join(origpath, "temp/labels"), val_idx_list, imgformat=".jpg")

def split_spndata(origpath, train_idxpath, val_idxpath):

    img_ext, label_ext = "image", "label"

    with open(train_idxpath) as f:
        reader = csv.reader(f)
        for line in reader:
            train_idx_list = line

    with open(val_idxpath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
           val_idx_list = row
        #val_idx_list = list(reader)

    print(len(train_idx_list), len(val_idx_list))

    fromfolder = "temp52"
    tofolder1 = "validation52"
    tofolder2 = "testing52"


    split_folder(os.path.join(origpath, f"{fromfolder}/states"), os.path.join(origpath, f"{tofolder1}/states"), train_idx_list, imgformat=".jpg")
    split_folder(os.path.join(origpath, f"{fromfolder}/labels"), os.path.join(origpath, f"{tofolder1}/labels"), train_idx_list, imgformat=".jpg")

    split_folder(os.path.join(origpath, f"{fromfolder}/states"), os.path.join(origpath, f"{tofolder2}/states"), val_idx_list, imgformat=".jpg")
    split_folder(os.path.join(origpath, f"{fromfolder}/labels"), os.path.join(origpath, f"{tofolder2}/labels"), val_idx_list, imgformat=".jpg")


def split_folder(origpath, destpath, idx_list, imgformat=".png"):
    """   Copy items defined by the idx_list from origpath to destpath.   """

    # check duplicates
    if len(idx_list) == len(set(idx_list)):
        print("nodupl")
    else:
        print("dupl")

    file = os.listdir(origpath)[0]
    if file.endswith(".txt"): adding = ".txt"
    elif file.endswith(imgformat): adding = imgformat
    else: print("Error")

    cnt = 0
    for idx in idx_list:
            src = os.path.join(origpath, str(idx) + adding)
            if os.path.exists(src): cnt += 1
            dst = os.path.join(destpath, str(idx) + adding)
            shutil.copy(src, dst)

    print(cnt)



def get_files2split(dpath, spath, ratio, batch_size):
    """
    Split a dataset into two given a ratio. Returns only the file names, doesn't perform the split.
    Makes sure the number of instances in both datasets are divisible by batch_size.
    :param dpath: path of the images or labels folder
    :param spath: path of the folder where the output file list should be saved
    :param ratio: split ratio of the first dataset
    :return: list of the filenames in the resulting datasets
    """

    files = [x for x in os.listdir(dpath) if (x.endswith(".txt") or x.endswith(".png"))]

    # Make sure the split is divisible by the batch size
    n1 = int(np.round((len(files) * ratio) / batch_size) * batch_size)
    n2 = int(np.floor((len(files) - n1) / batch_size) * batch_size)
    print(n1, n2)

    list1 = random.sample(files, k=n1)
    temp = [x for x in files if x not in list1]
    list2 = random.sample(temp, k=n2)

    # Remove extensions from file names
    list1noext = [x.replace(".png", "").replace(".txt", "") for x in list1]
    list2noext = [x.replace(".png", "").replace(".txt", "") for x in list2]

    print(f"List lenghts {len(list1noext)}, {len(list2noext)}")


    with open(os.path.join(spath, "train_ids.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(list1noext)
        #f.write(str(list1noext).replace("[", "").replace("]",""))

    with open(os.path.join(spath, "valid_ids.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(list2noext)
        #f.write(str(list1noext).replace("[", "").replace("]",""))


    return list1noext, list2noext


if __name__ == '__main__':

    """
    # Splitting PascalVoc
    labelpath = "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012/temp/labels"
    savepath = "/home/blanka/ERLPruning"
    #get_files2split(labelpath, savepath, 0.5, batch_size=16)

    origpath =  "/data/blanka/DATASETS/PascalVoc/VOCdevkit/VOC2012"
    train_idxpath = "/home/blanka/ERLPruning/PascalVoc_train_ids.csv"
    val_idxpath = "/home/blanka/ERLPruning/PascalVoc_valid_ids.csv"
    split_pascalvoc(origpath, train_idxpath, val_idxpath)
    """

    """
    # Splitting KITTI
    labelpath = '/data/blanka/DATASETS/KITTI/validation/label_3'
    savepath = "/home/blanka/ERLPruning"
    #get_files2split(labelpath, savepath, 0.5, batch_size=16)

    origpath = "/data/blanka/DATASETS/KITTI/original"
    train_idxpath = "/home/blanka/ERLPruning/KITTI2_valid_ids.csv"
    val_idxpath = "/home/blanka/ERLPruning/KITTI2_test_ids.csv"
    split_kitti(origpath, train_idxpath, val_idxpath)
    """

    # Splitting SPN data
    labelpath = '/data/blanka/DATASETS/SPN/temp52/labels'
    savepath = "/home/blanka/ERLPruning"
    # get_files2split(labelpath, savepath, 0.5, batch_size=1024)

    origpath = "/data/blanka/DATASETS/SPN"
    train_idxpath = "/home/blanka/ERLPruning/spn52_valid_ids.csv"
    val_idxpath = "/home/blanka/ERLPruning/spn52_test_ids.csv"
    split_spndata(origpath, train_idxpath, val_idxpath)
