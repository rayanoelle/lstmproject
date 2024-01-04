import pandas as pd
dataset = pd.read_csv('D:\\master\\05\\project\\data\\2017\\raw\\container_usage.csv')
dataset.columns=['timestamp','instance_id','cpu_util','mem_util','disk_util','avg_cpu_load_1','avg_cpu_load_5','avg_cpu_load_15','avg_cpi','avg_mpki','max_cpi','max_mpki']
dataset = dataset.sort_values(by="timestamp",ascending=True)
dataset['timestamp'] = pd.to_datetime(dataset['timestamp']+ 1483228800,unit='s')
dataset.fillna(method='pad', inplace=True)
dataset.to_csv('D:\\master\\05\\project\\data\\2017\\raw\\cleaned_container_usage.csv',index=False)
