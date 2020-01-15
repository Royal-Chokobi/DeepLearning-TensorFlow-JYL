import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np


stockItem = '060250'

url = 'http://finance.naver.com/item/sise_day.nhn?code='+ stockItem
html = urlopen(url)
source = BeautifulSoup(html.read(), "html.parser")

maxPage = source.find_all("table",align="center")
mp = maxPage[0].find_all("td",class_="pgRR")
mpNum = int(mp[0].a.get('href')[-3:])

finance_list = []

for page in range(mpNum, 0, -1):
    print (str(page) )
    url = 'http://finance.naver.com/item/sise_day.nhn?code=' + stockItem +'&page='+ str(page)
    html = urlopen(url)
    source = BeautifulSoup(html.read(), "html.parser")
    srlists=source.find_all("tr")
    isCheckNone = None

    # if((page % 1) == 0):
    #     time.sleep(1.50)

    for i in range(len(srlists)-1, 0, -1):
        if(srlists[i].span != isCheckNone):

            # srlists[i].td.text
            date = srlists[i].find_all("td",align="center")[0].text
            fin_date = int( srlists[i].find_all("td",align="center")[0].text.replace('.','') )
            adj_Close = float( srlists[i].find_all("td",class_="num")[0].text.replace(',','') )
            adj_won =  float( srlists[i].find_all("td",class_="num")[1].text.replace(',','') )
            if adj_won > 0:
                if str(srlists[i].find_all("td",class_="num")[1].find('img').get('alt')) == '하락':
                    adj_won = adj_won * -1
            open_pri = float( srlists[i].find_all("td",class_="num")[2].text.replace(',','') )
            high_pri = float( srlists[i].find_all("td",class_="num")[3].text.replace(',','') )
            low_pri = float( srlists[i].find_all("td",class_="num")[4].text.replace(',','') )
            adj_mount = float( srlists[i].find_all("td",class_="num")[5].text.replace(',','') )

            print("날짜 : {}, 종가 : {}, 전일비 : {}, 시가 : {}, 고가 : {}, 저가 : {}, 거래량 : {}".format(date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount))
            finance_list.append([date, fin_date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount])


finance_array = np.array(finance_list)
finance_df = pd.DataFrame(np.array(finance_array),columns=['realDate', 'Date', 'Adj_Close', 'Adj_Won', 'Open', 'High', 'Low', 'Volume'])
print(finance_df.head())

finance_df.to_csv("finance.csv", mode='w', header=True, index=False)