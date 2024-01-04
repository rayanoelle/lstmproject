
import yaml, json, re
from prometheus_api_client import PrometheusConnect
import pandas as pd
import numpy as np
import time
import math
import os
from tslearn.metrics import dtw
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from kubernetes import client , config


def metric_dict_to_df(metric, metric_name):
    
    df = pd.DataFrame(np.array(metric), columns=['timestamp' , metric_name])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp' , inplace= True)
    df[metric_name] =df[metric_name].astype(float)
    df.interpolate(limit_direction="both",inplace=True)
    return df

def statisticalColumns(df):
    '''
    adds statistical columns to data
    ma6 = movinga average with 6 past windows
    ma12 = movinga average with 12 past windows
    ema = exponential moving average
    '''
    df['ma6'] = df['cpu_util'].rolling(window = 6).mean()
    df['ma12'] = df['cpu_util'].rolling(window = 12).mean()
    df['ema'] = df['cpu_util'].ewm(com=0.7).mean()
    df.interpolate(limit_direction="both",inplace=True)
    df['ma6'].fillna((df['ma6'].mean()), inplace=True)
    df['ma6'].fillna((0), inplace=True)
    df['ma12'].fillna((df['ma12'].mean()), inplace=True)
    df['ma12'].fillna((0), inplace=True)
    return df

def getArimaHyperparamters(timeseries):
    '''
    returns differencing level for arima column
    '''
    values = timeseries.values
    iterration = 0
    p_values = []
    adfuller_result = adfuller(values)
    p_values.append(adfuller_result[1])
    while adfuller_result[1] > 0.5 and iterration <=5 :
        timeseries = timeseries.diff().fillna(method='ffill').fillna(method='bfill')
        values= timeseries.values
        adfuller_result = adfuller(values)
        p_values.append(adfuller_result[1])
        iterration += 1
    min_value = min(p_values)
    arima_d = p_values.index(min_value)   
    if arima_d == 0:
        arima_d +=1
    return arima_d 

def arimaModel(timeseries_data):
    ''' 
    creates arima predictions for the 48 last lines of the dataframe
    '''
    # arima_d = getArimaHyperparamters(timeseries_data)
    timeseries_data = timeseries_data.values
    size = int(len(timeseries_data) * 0.66)
    train, test = timeseries_data[0:size], timeseries_data[size:len(timeseries_data)]
    history = [x for x in train]
    predictions = list()
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(6,arima_d,0))
    #     model.initialize_approximate_diffuse()
    #     model_fit = model.fit()
    #     output = model_fit.forecast()
    #     print(output)
    #     yhat = output[0]
    #     predictions.append(yhat)
    #     obs = test[t]
    #     history.append(obs)
    # fit model
    # model = ARIMA(train, order=(7,1,0))
    # model_fit = model.fit()
    # one-step out of sample forecast
    # start_index = len(train)
    # end_index = len(timeseries_data)
    # predictions = model_fit.predict(start=start_index, end=end_index)
    predictions = timeseries_data
    arima_prediction = pd.DataFrame(predictions)
    return arima_prediction 

def arimaColumn(df):
    '''
    adds arima predictions as column to our dataset
    '''
    cpu_util_series = df['cpu_util']
    arima_prediction = arimaModel(cpu_util_series)
    df.loc[df.index[-len(arima_prediction):],'ARIMA'] = arima_prediction.values
    df.loc[df.index[:-len(arima_prediction)], 'ARIMA'] = cpu_util_series[:-len(arima_prediction)]
    return df

def lagFeatures(future_steps, past_steps, data):
    input_features =[]
    target_features= []
    # Reformat input data into a shape: (n_samples x timesteps x n_features)
    # In my example, my df_for_training_scaled has a shape (12823, 5)
    # 12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(past_steps, len(data) - future_steps + 1):
        input_features.append(data[i - past_steps:i, :])
        # target_features.append(data[i + future_steps - 1:i + future_steps, 0])
    return input_features

def inverseData(data, data_shape, scaler):
    ''' 
    inverting our data for evaluation
    '''
    data_copies = np.repeat(data,data_shape[1], axis = -1)
    inverse_data = scaler.inverse_transform(data_copies)[:,0]
    return inverse_data

config.load_kube_config()
api_client = client.ApiClient()
apps_v1 = client.AppsV1Api(api_client)
configs = None
with open("config.yaml", "r") as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
barycenters = {}
for filename in os.listdir(configs['barycenter_path']):
    with open(configs['barycenter_path']+filename,'r') as file:
        barycenters[filename] = []
        for line in file:
            barycenters[filename].append(float(line))

models= {}
model_size = {}
for cluster_file in os.listdir(configs['model_path']):
    json_file = open("%s/%s/model.json" % (configs['model_path'], cluster_file), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_config= json.loads(loaded_model_json)
    for i in range(len(model_config["config"]["layers"])):
        if model_config["config"]["layers"][i]["class_name"] == "InputLayer":
            model_size[cluster_file] = model_config["config"]["layers"][i]["config"]["batch_input_shape"][1]
    models[cluster_file]= model_from_json(loaded_model_json)
    # load weights into new model
    # models[cluster_file].load_weights("%s/%s/model.h5" % (config['model_path'], cluster_file))
desired_cpu_usage_lower_limit = 30
desired_cpu_usage_upper_limit = 50
v1 = client.CoreV1Api()
pod_list = v1.list_namespaced_pod("default")
metric_collector_pods = ['grafana-695b9bb8bc-b5n8r',
                         'prometheus-alertmanager-0',
                         'prometheus-kube-state-metrics-7f6769f7c6-gxnnh',
                         'prometheus-prometheus-node-exporter-klxnz',
                         'prometheus-prometheus-pushgateway-684dc6674-7n5d5',
                         'prometheus-server-b9bdb5877-g8jp9']

while True:
    pod_list = v1.list_namespaced_pod("default")
    current_pod_count = 0
    for pod in pod_list.items:
        if pod.status.phase == 'Running':
            if pod.metadata.name not in metric_collector_pods:
                current_pod_count +=1

    prom = PrometheusConnect(url =configs['prometheus_address'], disable_ssl=True)

    cpu_load_avg_1 = prom.custom_query(
        """(avg(rate(container_cpu_load_average_10s{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}[2m])) by(pod))[1h:5m]"""
    )

    cpu_load_avg_5 = prom.custom_query(
        """(avg(rate(container_cpu_load_average_10s{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}[5m])) by(pod))[1h:5m]"""
    )

    cpu_load_avg_15 = prom.custom_query(
        """(avg(rate(container_cpu_load_average_10s{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}[15m])) by(pod))[1h:5m]"""
    )

    memory_usage_percentage = prom.custom_query(
        """((avg(container_memory_working_set_bytes{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}) by (pod) / avg(container_spec_memory_limit_bytes{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}) by (pod))*100)[1h:5m]""")

    cpu_usage_percentage = prom.custom_query(
        """(sum(rate(container_cpu_usage_seconds_total{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}[5m])) by (pod) / sum(container_spec_cpu_quota{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}/container_spec_cpu_period{namespace="default",pod!~"prometheus.*|kube.*|grafana.*|nginx-deployment-774f96d4d9-jq6lz|nginx-deployment-774f96d4d9-7tmfm|pod1"}) by (pod) * 100)[1h:5m]"""
    )
    deployment_pod_regex = r'^([a-zA-Z0-9\-_]*)\-[a-zA-Z0-9]{3,}\-[a-zA-Z0-9]+$'
    deployments = {}
    for metric in cpu_usage_percentage:
        if "pod" in metric["metric"]:
            label = None
            if m := re.match(deployment_pod_regex, metric["metric"]["pod"]):
                label =  m.group(1)
                if label not in deployments:
                    deployments[label] = {"pods": {"cpu_usage_percentage" : [],
                                                "memory_usage_percentage" : [],
                                                "cpu_load_avg_1" : [],
                                                "cpu_load_avg_5" : [],
                                                "cpu_load_avg_15" : []}}
                deployments[label]["pods"]["cpu_usage_percentage"].append(metric)

    for metric in memory_usage_percentage:
        if "pod" in metric["metric"]:
            label = None
            if m := re.match(deployment_pod_regex, metric["metric"]["pod"]):
                label =  m.group(1)
                deployments[label]["pods"]["memory_usage_percentage"].append(metric)

    for metric in cpu_load_avg_1:
        if "pod" in metric["metric"]:
            label = None
            if m := re.match(deployment_pod_regex, metric["metric"]["pod"]):
                label =  m.group(1)
                deployments[label]["pods"]["cpu_load_avg_1"].append(metric)

    for metric in cpu_load_avg_5:
        if "pod" in metric["metric"]:
            label = None
            if m := re.match(deployment_pod_regex, metric["metric"]["pod"]):
                label =  m.group(1)
                deployments[label]["pods"]["cpu_load_avg_5"].append(metric)

    for metric in cpu_load_avg_15:
        if "pod" in metric["metric"]:
            label = None
            if m := re.match(deployment_pod_regex, metric["metric"]["pod"]):
                label =  m.group(1)
                deployments[label]["pods"]["cpu_load_avg_15"].append(metric)
    # print(json.dumps(cpu_usage_percentage, sort_keys=True, indent=4))

    # print(json.dumps(deployments, sort_keys=True, indent=4))
    for g in deployments:
        avg = {}
        for pod in range(len(deployments[g]["pods"]["cpu_usage_percentage"])):
            for d in deployments[g]["pods"]["cpu_usage_percentage"][pod]["values"]:
                if d[0] not in avg:
                    avg[d[0]] = {'count': 0, 'sum': float(0)}
                avg[d[0]]['count'] += 1
                avg[d[0]]['sum'] += float(d[1])
                
        for a in avg:
            avg[a] = avg[a]['sum'] / avg[a]['count']
        
        deployments[g]['avg'] = []

        for k in sorted(avg):
            deployments[g]['avg'].append(avg[k])

        deployment_cluster = None
        deployment_dtw = float('inf')
        for cluster in barycenters:
            temp_dtw = dtw(deployments[g]['avg'],barycenters[cluster] )
            if temp_dtw < deployment_dtw:
                deployment_dtw = temp_dtw
                deployment_cluster = cluster
        deployments[g]['cluster'] = deployment_cluster
        deployment_pods = []
        current_cpu_usage = deployments[g]["avg"][-1]
        metric_collector_pod_count = len(deployments[g]["pods"]["cpu_usage_percentage"])
        for pod in range(metric_collector_pod_count):
            pod_name = deployments[g]["pods"]["cpu_usage_percentage"][pod]["metric"]["pod"]
            cpu_usage_percentage_df = metric_dict_to_df(deployments[g]["pods"]["cpu_usage_percentage"][pod]["values"],"cpu_util")
            for mem_pod in range(metric_collector_pod_count):
                if np.array(deployments[g]["pods"]["memory_usage_percentage"][pod]["values"]).size != 0:
                    if  deployments[g]["pods"]["memory_usage_percentage"][mem_pod]["metric"]["pod"] == pod_name:
                        memory_usage_percentage_df = metric_dict_to_df(deployments[g]["pods"]["memory_usage_percentage"][pod]["values"],"memory_usage_percentage")
                else:
                    if  deployments[g]["pods"]["memory_usage_percentage"][mem_pod]["metric"]["pod"] == pod_name:
                        memory_usage_percentage_df = metric_dict_to_df([time.time(),0],"memory_usage_percentage")

            for load_1 in range(metric_collector_pod_count):
                if np.array(deployments[g]["pods"]["cpu_load_avg_1"][pod]["values"]).size != 0:
                    if  deployments[g]["pods"]["cpu_load_avg_1"][load_1]["metric"]["pod"] == pod_name:
                        cpu_load_avg_1_df = metric_dict_to_df(deployments[g]["pods"]["cpu_load_avg_1"][pod]["values"],"cpu_load_avg_1")
                else:
                    if  deployments[g]["pods"]["cpu_load_avg_1"][load_1]["metric"]["pod"] == pod_name:
                        cpu_load_avg_1_df = metric_dict_to_df([time.time(),0],"cpu_load_avg_1")
            
            for load_5 in range(metric_collector_pod_count):
                if np.array(deployments[g]["pods"]["cpu_load_avg_5"][pod]["values"]).size != 0:
                    if  deployments[g]["pods"]["cpu_load_avg_5"][load_5]["metric"]["pod"] == pod_name:
                        cpu_load_avg_5_df = metric_dict_to_df(deployments[g]["pods"]["cpu_load_avg_5"][pod]["values"],"cpu_load_avg_5")
                else:
                    if  deployments[g]["pods"]["cpu_load_avg_15"][load_5]["metric"]["pod"] == pod_name:
                        cpu_load_avg_5_df = metric_dict_to_df([time.time(),0],"cpu_load_avg_5")
            
            for load_15 in range(metric_collector_pod_count):
                if np.array(deployments[g]["pods"]["cpu_load_avg_15"][pod]["values"]).size != 0:
                    if  deployments[g]["pods"]["cpu_load_avg_15"][load_15]["metric"]["pod"] == pod_name:
                        cpu_load_avg_15_df = metric_dict_to_df(deployments[g]["pods"]["cpu_load_avg_15"][pod]["values"],"cpu_load_avg_15")
                else:
                    if  deployments[g]["pods"]["cpu_load_avg_15"][load_15]["metric"]["pod"] == pod_name:
                        cpu_load_avg_15_df = metric_dict_to_df([time.time(),0],"cpu_load_avg_15")

            metric_df = pd.concat([cpu_usage_percentage_df , memory_usage_percentage_df , cpu_load_avg_1_df , cpu_load_avg_5_df , cpu_load_avg_15_df] , axis= 1)
            metric_df.interpolate(limit_direction="both",inplace=True)
            metric_df = statisticalColumns(metric_df)
            metric_df = arimaColumn(metric_df)
            deployment_pods.append(metric_df)
            

        if len(deployment_pods) > 1 :
            dataset = pd.concat(deployment_pods)
        else:
            dataset = deployment_pods[0]
        
        models[deployments[g]['cluster'][:-4]].load_weights("%s%s/model.h5" % (configs['model_path'], deployments[g]['cluster'][:-4]))
        values = dataset.values
        # normalize features
        # we fit and transform in differet lines for inversing transformation in evaluation section
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
        scaled = scaler.transform(values)
        # Number of days we want to look into the future based on the past days.
        future = 1
        # Number of past days we want to use to predict the future.
        past = model_size[deployments[g]['cluster'][:-4]]
        #producing lag features 
        input_features  = lagFeatures(future_steps=future, past_steps= past,data=scaled)
        input_features = np.array(input_features)

        deployment_prediction = models[deployments[g]['cluster'][:-4]].predict(input_features[-future:])
        shape = values.shape
        cpu_usage_prediction = float(inverseData(deployment_prediction,shape, scaler)[0])*10
        print(cpu_usage_prediction)
        desired_pod_count = None
        if cpu_usage_prediction < 30 and current_pod_count > 1 :
            print('downscale condition')
            desired_pod_count = math.ceil(current_pod_count *(cpu_usage_prediction/desired_cpu_usage_lower_limit))
            print(desired_pod_count,current_pod_count)
        elif cpu_usage_prediction > 50:
            print('upscale condition')
            desired_pod_count = math.ceil(current_pod_count *(cpu_usage_prediction/desired_cpu_usage_upper_limit))
            print(desired_pod_count,current_pod_count)
        else:
            continue
        api_response = apps_v1.patch_namespaced_deployment_scale(str(g), "default", {"spec": {"replicas": desired_pod_count}})
    time.sleep(30)
