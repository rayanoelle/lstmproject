from kubernetes import client , config
config.load_kube_config()
api_client = client.ApiClient()
apps_v1 = client.AppsV1Api(api_client)
v1 = client.CoreV1Api()
pod_list = v1.list_namespaced_pod("default")
pod_list = v1.list_namespaced_pod("default")
current_pod_count = 0
metric_collector_pods = ['grafana-695b9bb8bc-b5n8r',
                         'prometheus-alertmanager-0',
                         'prometheus-kube-state-metrics-7f6769f7c6-gxnnh',
                         'prometheus-prometheus-node-exporter-klxnz',
                         'prometheus-prometheus-pushgateway-684dc6674-7n5d5',
                         'prometheus-server-b9bdb5877-g8jp9'
                        #  'nginx-deployment-774f96d4d9-jq6lz',
                        #  'nginx-deployment-774f96d4d9-7tmfm'
                        ]
for pod in pod_list.items:
    if pod.status.phase == 'Running':
        if pod.metadata.name not in metric_collector_pods:
            current_pod_count +=1
            print(pod.metadata.name)
