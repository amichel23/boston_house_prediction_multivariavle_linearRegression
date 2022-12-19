from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston = load_boston()
data = pd.DataFrame(data=boston.data, columns = boston.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

response = np.log(boston.target)
response = pd.DataFrame(response, columns=['PRICE'])
response.shape

CRIME_IDX = 0
ZN_IDX = 1
CHAZ_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8


property_stats = features.mean().values.reshape(1,11)

lin_reg = LinearRegression().fit(features, response)
fitted_vals = lin_reg.predict(features)

MSE = round(mean_squared_error(response, fitted_vals),3)
RMSE = round(np.sqrt(MSE),3)

# function that takes input and returns prediction, upper & lower bounds, and confidence interval
def get_log_estimate(num_rooms,
                    stud_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    
    #configure property
    property_stats[0][RM_IDX] = num_rooms
    property_stats[0][PTRATIO_IDX] = stud_per_classroom
    
    if next_to_river:
        property_stats[0][CHAZ_IDX] = 1
    else:
        property_stats[0][CHAZ_IDX] = 0
    
    #make prediction
    log_estimate = lin_reg.predict(property_stats)[0][0]
    
    # calc range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    
    
    return log_estimate, upper_bound, lower_bound, interval


def dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """
    Function to take in the get_log_estimate, do conversions, and return prediction with upper/lower/intervals
    Keyword arguements:
    rm -- number of rooms
    ptratio -- number of students per teacher in the classroom for the school in the area
    chas -- boolean, on the river or not
    large_range -- boolean, 68 or 95 confidence interval
    """
    if rm < 1 or ptratio < 1:
        print('Impossible. Try again.')
        return
    
    log_est, upper, lower, conf = get_log_estimate(num_rooms=rm, stud_per_classroom=ptratio, next_to_river=chas, high_confidence=large_range)
    REALTOR_MEDIAN_PRICE = 799.0
    SCALE_FACTOR = REALTOR_MEDIAN_PRICE / np.median(boston.target)
    # convert to today's dollars

    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR

    # round to nearest thousand

    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')
