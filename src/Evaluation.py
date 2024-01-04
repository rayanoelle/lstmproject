'''
this class is used for reporting prediction error in 
'''
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import math

class Evaluation:
    def __init__(self):
        self.metrics = dict()
    
    def get_evaluation(self,original_data, prediction,status):
        mape = mean_absolute_percentage_error(original_data,prediction)*100
        rmse = math.sqrt(mean_squared_error(original_data,prediction))
        metrics ={
            "status": status,
            "mean_absolute_percentage_error": mape,
            "root_mean_squared_error": rmse
        }
        return metrics
    