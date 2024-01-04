'''
goal : seperate data for each unique container
input: container usage csv
output: seperate individual container csv
'''

# native libraries for working with files
import os
import math
#libraries for working with datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#loading dataset
dataset = pd.read_csv('D:\\master\\05\\project\\data\\2017\\raw\\cleaned_container_usage.csv', header=0,index_col=0)

#list of unique container ids
container_id_list = pd.unique(dataset['instance_id'])

#list of seperate data
containers = list()
column_names=list()
for container_id in container_id_list:
    container_data_holder = dataset[dataset['instance_id'] == container_id]
    if len(container_data_holder) > 100:
        containers.append(container_data_holder)
        column_names += ['container_%s' %container_id]

#making sure different data have similar length
# this is necessary for clustering and prediction purposes

#find series with maximum length
series_lengths = {len(series) for series in containers}
max_len = max(series_lengths)
longest_series = None
for series in containers:
    if len(series) == max_len:
        longest_series = series
#find containers whit smaller size
problems_index = []

for i in range(len(containers)):
    if len(containers[i]) != max_len:
        problems_index.append(i)
        containers[i] = containers[i].reindex(longest_series.index)

#using interpolate to fill nan values

for i in problems_index:
    containers[i].interpolate(limit_direction="both",inplace=True)

#writing data into seperate csv
for i in range(0 , len(column_names)):
    containers[i].to_csv('D:\\master\\05\\project\\data\\2017\\seperate_datasets\\'+str(column_names[i])+'.csv')
