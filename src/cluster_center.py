from tslearn.metrics import dtw
import csv
import os
directory = '../data/clusters/'
cluster_directoy = '../data/engineered_data/'
barycenter_directory = '../data/barycenters/'
for filename in os.listdir(directory):
    count = 0
    summation = []
    if filename.endswith(".csv"):
        with open(directory+filename, 'r') as cluster_file:
            cluster_reader = csv.reader(cluster_file)
            for container_name in cluster_reader:
                if container_name[1] ==  'Series':
                    continue
                with open(cluster_directoy+container_name[1]+'.csv' , 'r') as container_file:
                    container_reader = csv.reader(container_file)
                    i = 0
                    for feature in container_reader:
                        if feature[1] == 'cpu_util':
                            continue
                        if count == 0:
                            summation.append(float(feature[1]))
                        else:
                            summation[i] += float(feature[1])
                        i += 1
                count += 1
    with open(barycenter_directory+filename ,'w') as file:
        for s in summation:
            
            file.write("%f\n" % (s/count))

                
                