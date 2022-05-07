import numpy as np
import pandas as pd

import re
import os
import csv
import glob

def crt_dtl_csv(set, images):
    with open(set+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for imgs in images:
            fn = os.path.basename(imgs)
            label = re.search('(?<=path_)[0-9]+', fn).group(0)
            spamwriter.writerow([fn, label])

train_dataset_directory = './dataset/train'
test_dataset_directory = './dataset/test'

train_images = glob.glob(train_dataset_directory+'/*.png')
test_images = glob.glob(test_dataset_directory+'/*.png')


if __name__ == "__main__":
	crt_dtl_csv('train', train_images)
	crt_dtl_csv('test', test_images)
