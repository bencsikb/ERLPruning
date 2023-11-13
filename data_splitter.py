import argparse
import numpy as np
import random
import csv

""" To run without import issues, move this file to the root folder. """


def split_data(dpath, spath, batch_size, ratio, name_ext=''):
    """
    Splits the data into train, val, test in the given ratio.
    Saves the selected filenames in .txt files. 
    Inputs:
        - dpath (str): path to all/states data folder.
        - spath (str): path to the folder where the .csv file should be saved (e.x. data/IDsForSplit)
        - ratio (list): train and validation ratio (test is the rest) 
        - name_ext (str): name of the split
        
    """
    
    files = [x for x in os.listdir(dpath) if (x.endswith(".txt") or x.endswith(".png"))]

    # Make sure the split is divisible by the batch size
    n1 = int(np.round((len(files) * ratio[0]) / batch_size) * batch_size)
    n2 = int(np.round((len(files) * ratio[1]) / batch_size) * batch_size)
    print(n1, n2)

    list1 = random.sample(files, k=n1)
    temp = [x for x in files if x not in list1]
    list2 = random.sample(temp, k=n2)
    list3 = [x for x in temp if x not in list2]

    # Remove extensions from file names
    list1noext = [x.replace(".png", "").replace(".txt", "") for x in list1]
    list2noext = [x.replace(".png", "").replace(".txt", "") for x in list2]
    list3noext = [x.replace(".png", "").replace(".txt", "") for x in list3]


    print(f"List lenghts {len(list1noext)}, {len(list2noext)}, {len(list3noext)}")

    with open(os.path.join(spath, f"{name_ext}_train.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(list1noext)

    with open(os.path.join(spath, f"{name_ext}_valid.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(list2noext)
        
    with open(os.path.join(spath, f"{name_ext}_test.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(list3noext)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', default="/nas/blanka_phd/DATASETS/SPN/COCO/all/states")
    parser.add_argument('--spath', default="data/IDsForSplit")    
    parser.add_argument('--batch_size', default=4096)    
    parser.add_argument('--name_ext', default="spn_dsg_01_") 
    opt = parser.parse_args()
    
    split_data(opt.dpath, opt.spath, opt.batch_size, opt.name_ext)

