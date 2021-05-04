# This file will serve to create a CSV that contains all lotto results since Draw #1.


import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
import csv
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'}

# The number of draws must firstly be calculated.
# A draw occurs every Wednesday and Saturday; 2 times a week.
# The first draw occurred on July 4th, 2001.
# To account for anomalies or specific events where draws could not happen, a variable is assigned for margin of error.
# This can be changed as required.
error = 31

def getTotalDraws():
    d1 = date(2001, 7, 4)
    d2 = date.today()
    return (((d2-d1).days // 7)*2)-error


# Having gotten approximately the number of draws, the data must now be compiled to a CSV, for further data processing.
# The core link to query draws is stored in the query variable. This may need to be updated if the URL is every changed in the future.
# For every draw, the data is pulled, using bs4, and written into the CSV.
# Additional data may also be pulled, to assist the ML model later on (TODO).

# The createCSV() makes the following assumptions:
# - The webpage remains as it is on 3/5/2021.
# - The class names are as programmed, and are in the appropriate element (div).
# - The draw consists of 6 numbers, the last of which is considered the PowerBall.
# - The PowerBall has a different class than the other elements
# - The date for each draw is found within a <strong> elements.

classForDraws = "ball lotto-balls"
classForPowerBall = "ball lotto-balls yellow-ball"

def createCSV():
    maxQueries = getTotalDraws()
    query = "http://www.nlcbplaywhelotto.com/nlcb-lotto-plus-results/?drawnumber="
    cls = lambda: os.system('cls')
    with open('draws.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Draw Number', 'Date', 'Jackpot', '1', '2', '3', '4','5', 'PB'])
        for i in range(1, maxQueries+1):
            # In case any error occurs, it just skips that iteration. 
            # It was discovered that some of the webpages do not contain the jackpot amount. Ignoring these would result in a smaller dataset,
            # but I am trying to have as much features to use, with one being the jackpot amount.
            try:
                print(i, "/", maxQueries)
                nums = []
                pb = ""
                drawDate = ""
                drawAmt = ""
                queryTemp = query + str(i)
                r = requests.get(queryTemp, headers=headers)
                soup = BeautifulSoup(r.text, 'lxml')
                result1 = soup.find_all('div', class_= classForDraws)
                for num in result1:
                    if num is None:
                        nums.append('None')
                    else:
                        nums.append(num.text)
                result2 = soup.find('div', class_= classForPowerBall)
                if result2 is None:
                    pb = 'None'
                else:
                    pb = result2.text

                result3 = soup.find_all('strong')[3].text.split(' ')[1]
                drawDate = result3

                result4 = float(soup.find('div', class_="drawDetails").text.split()[9].rstrip('Number\n').lstrip('\n$').replace(',',''))
                drawAmt = result4

                # Write into to CSV
                writer.writerow([str(i), drawDate, drawAmt, nums[0], nums[1], nums[2], nums[3], nums[4], pb])
                cls()
            
            except:
                print("Skipping...")
                continue
            
createCSV()

