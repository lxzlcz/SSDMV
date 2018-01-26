# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data
from LadderClass import Ladder
import math
import os
import csv
import numpy as np
from tqdm import tqdm

num_epochs = 5
num_labeled = 200

print "===  Loading Data ==="
for i in range(5):
    data = input_data.read_data_sets_views(n_labeled=num_labeled, TEST_SIZE=8000)
    num_examples = data.train.num_examples
    ladder = Ladder(label_nums=num_labeled)
    ladder.train(num_examples,data,num_epochs)