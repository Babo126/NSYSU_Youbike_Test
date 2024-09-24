import requests
import pandas as pd
from datetime import datetime
from pprint import pprint
import json
import time
import datetime
import csv
import os

#透過註冊TDX會員，獲取以下必要資料
client_id = 'b107610021-638e226a-b559-48c6'
client_secret = 'c3f04fd9-faef-46c9-b5f9-293c2d41e239'

token_url="https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
base_url = "https://tdx.transportdata.tw/api/advanced/v2/Bike"
availability_endpoint = "/Availability/NearBy"
station_endpoint = "/Station/NearBy"
# 前30組資料
top = "30"  
#國立中山大學幾何中心座標：
lat = "22.62752590909029"
lng = "120.26465291318681"
#半徑至多1000公尺
radius = "1000"
AVAILABILITY_URL = f"{base_url}{availability_endpoint}?%24top={top}&%24spatialFilter=nearby%28{lat}%2C%20{lng}%2C%20{radius}%29&%24format=JSON"
STATION_URL = f"{base_url}{station_endpoint}?%24top={top}&%24spatialFilter=nearby%28{lat}%2C%20{lng}%2C%20{radius}%29&%24format=JSON"

class TDX():

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret

    def get_token(self):
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(token_url, headers=headers, data=data)
        print("response.status_code:%d" % response.status_code)
        # print(response.json())
        return response.json()['access_token']

    def get_response(self, url):
        headers = {'authorization': f'Bearer {self.get_token()}'}
        response = requests.get(url, headers=headers)
        return response.json()

if __name__ == '__main__':
    while True:
            try:
                tdx = TDX(client_id, client_secret)
                availability_response = tdx.get_response(AVAILABILITY_URL)
                station_response = tdx.get_response(STATION_URL)
                
                # 創建一個空的 DataFrame 用於存放比對後的資料
                df_result = pd.DataFrame()

                # 迭代 STATION_URL 的每一筆資料
                for station in station_response:
                    station_id = station['StationID']
                    station_name = station['StationName']['Zh_tw']
                    station_address = station['StationAddress']['Zh_tw']
                    bikes_capacity = station['BikesCapacity']

                    # 從 AVAILABILITY_URL 找出符合 StationID 的資料
                    for avail in availability_response:
                        if avail['StationID'] == station_id:
                            available_rent_bikes = avail['AvailableRentBikes']
                            available_rent_electricbikes = avail['AvailableRentBikesDetail']['ElectricBikes']
                            available_rent_generalbikes = avail['AvailableRentBikesDetail']['GeneralBikes']
                            available_return_bikes = avail['AvailableReturnBikes']
                            update_time = avail['UpdateTime']
                            break

                    # 將比對後的資料加入 DataFrame
                    new_row = pd.DataFrame([{
                        'StationID': station_id,
                        'StationName': station_name,
                        'StationAddress': station_address,
                        'BikesCapacity': bikes_capacity,
                        'AvailableRentBikes': available_rent_bikes,
                        'ElectricBikes': available_rent_electricbikes,
                        'GeneralBikes': available_rent_generalbikes,
                        'AvailableReturnBikes': available_return_bikes,
                        'UpdateTime': update_time
                    }])

                    df_result = pd.concat([df_result, new_row], ignore_index=True)

                print("國立中山大學幾何中心周圍1公里Youbike站點即時狀態:")
                print(df_result)
                # 設定要寫入的 CSV 檔名與路徑
                csv_path = '國立中山大學幾何中心周圍1公里Youbike站點即時狀態.csv'
                if os.path.isfile(csv_path):# 如果檔案已存在，就以 "a" 的模式打開 (append mode)
                    df_result.to_csv(csv_path, mode='a', index=False, encoding='utf-8-sig')
                else:# 如果檔案不存在，就創建一個新檔案並寫入欄位名稱
                    df_result.to_csv(csv_path, mode='a', index=True, encoding='utf-8-sig')
                print(f"Data collected at {datetime.datetime.now()}")
                time.sleep(600)
            except Exception as e:
                # 印出錯誤訊息
                print(f"Error occurred: {e}")
                time.sleep(600)
    

        

