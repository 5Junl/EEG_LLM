import os
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F


source_folder = r"E:\EEG_preprocessed\processed_3"
target_folder = r"E:\EEG_preprocessed\train_val_test_3"
train_folder = os.path.join(target_folder, 'train')
val_folder = os.path.join(target_folder, 'val')
test_folder = os.path.join(target_folder, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

source_list = os.listdir(source_folder)
train = os.listdir(train_folder)
val = os.listdir(val_folder)
test = os.listdir(test_folder)


def move(file_list, num_files_to_select, judge1, judge2, destination_folder):
    choose = [i for i in file_list if i not in judge1 and i not in judge2]

    selected_files = random.sample(choose, num_files_to_select)

    for file_name in selected_files:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy(source_file, destination_file)

    print(f'{num_files_to_select} files have been randomly selected and copied to the destination folder.')


move(source_list, 1120, judge1=val, judge2=test, destination_folder=train_folder)
move(source_list, 160, judge1=val, judge2=test, destination_folder=val_folder)
move(source_list, 320, judge1=val, judge2=test, destination_folder=test_folder)
