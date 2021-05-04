import predictor
import pandas as pd
import numpy as np

# Enter the current draw, date and amount here for a prediction.
drawNo = 2038
drawDate = '5-May-21'
amt = 2300000

drawDate = pd.to_datetime(drawDate, format='%d-%b-%y')
drawDate = (drawDate - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

print(predictor.classifier.predict(np.array([drawNo, drawDate, amt]).reshape(1,-1)))

