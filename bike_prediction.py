import pandas as pd
import os
import requests
import time
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)

client_id = 'b107610021-638e226a-b559-48c6'
client_secret = 'c3f04fd9-faef-46c9-b5f9-293c2d41e239'
token_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
base_url = "https://tdx.transportdata.tw/api/advanced/v2/Bike"
availability_endpoint = "/Availability/NearBy"
station_endpoint = "/Station/NearBy"
top = "30"
lat = "22.62752590909029"
lng = "120.26465291318681"
radius = "1000"
AVAILABILITY_URL = f"{base_url}{availability_endpoint}?%24top={top}&%24spatialFilter=nearby%28{lat}%2C%20{lng}%2C%20{radius}%29&%24format=JSON"
STATION_URL = f"{base_url}{station_endpoint}?%24top={top}&%24spatialFilter=nearby%28{lat}%2C%20{lng}%2C%20{radius}%29&%24format=JSON"

# 排除的站點ID列表，確認為字串格式
excluded_stations = ['501209024', '501209036', '501209042', '501209062']

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
        return response.json()['access_token']

    def get_response(self, url):
        headers = {'authorization': f'Bearer {self.get_token()}'}
        response = requests.get(url, headers=headers)
        return response.json()

def collect_data():
    try:
        tdx = TDX(client_id, client_secret)
        availability_response = tdx.get_response(AVAILABILITY_URL)
        station_response = tdx.get_response(STATION_URL)
        
        df_result = pd.DataFrame()
        
        for station in station_response:
            try:
                station_id = station['StationID']
                
                # 檢查該站點是否在排除的列表中
                if station_id in excluded_stations:
                    continue  # 跳過這個站點
                else:
                    station_name = station['StationName']['Zh_tw']
                    station_address = station['StationAddress']['Zh_tw']
                    bikes_capacity = station['BikesCapacity']
                    
                    for avail in availability_response:
                        if avail['StationID'] == station_id:
                            available_rent_bikes = avail['AvailableRentBikes']
                            available_rent_electricbikes = avail['AvailableRentBikesDetail']['ElectricBikes']
                            available_rent_generalbikes = avail['AvailableRentBikesDetail']['GeneralBikes']
                            available_return_bikes = avail['AvailableReturnBikes']
                            update_time = avail['UpdateTime']
                            break
                    
                    # 將數據加入到 DataFrame
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
            except KeyError as e:
                print(f"KeyError: {e} for station {station}")
            except TypeError as e:
                print(f"TypeError: {e} - possible string issue with {station}")
        
        # 指定 CSV 路徑，並確保目錄存在
        directory = './data'
        csv_path = os.path.join(directory, '國立中山大學幾何中心周圍1公里Youbike站點即時狀態.csv')

        # 如果目錄不存在，創建目錄
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 檢查檔案是否已存在，決定是新增還是創建
        if os.path.isfile(csv_path):
            df_result.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_result.to_csv(csv_path, mode='w', index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"Error occurred: {e}")

def predict_bike_availability_with_arima(data):
    low_bike_stations = []

    unique_stations = data['StationID'].unique()

    for station in unique_stations:
        station_data = data[data['StationID'] == station].reset_index(drop=True)

        if len(station_data) < 5:
            print(f"Skipping station {station} due to insufficient data.")
            continue
        
        # 使用 ARIMA 模型預測可借自行車數量
        model_rent = ARIMA(station_data['AvailableRentBikes'], order=(1,1,0))
        model_rent_fit = model_rent.fit()
        forecast_rent = model_rent_fit.forecast(steps=1)

        # 預測的可借車輛數除以總車位數量，得到可借車比例
        bikes_capacity = station_data['BikesCapacity'].iloc[0]
        predicted_avail_bikes = int(forecast_rent.iloc[0])  # 無條件捨去可借車輛數
        bike_availability_rate = predicted_avail_bikes / bikes_capacity

        # 將可借車比例加入到列表中
        low_bike_stations.append({
            'StationName': station_data['StationName'].iloc[0],
            'PredictedAvailableBikes': predicted_avail_bikes,
            'BikeAvailabilityRate': bike_availability_rate,
            'BikesCapacity': bikes_capacity
        })

    return low_bike_stations


if __name__ == "__main__":

    while True:
        collect_data()
        csv_path = './data/國立中山大學幾何中心周圍1公里Youbike站點即時狀態.csv'

        # 檢查檔案是否存在且不為空
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            data = pd.read_csv(csv_path)
        else:
            print("Data file not found or is empty, waiting for more data...")
            time.sleep(600)
            continue

        if len(data) < 50:
            print("Waiting for more data...")
            time.sleep(600)
            continue
"""
        low_bike_stations = predict_bike_availability_with_arima(data)

        print("Stations predicted to have the following bike availability in the next 10 mins:")
        for station in low_bike_stations:
            if station['BikeAvailabilityRate'] < 0.2:
                print(f"{station['StationName']}: Predicted Available Bikes = {station['PredictedAvailableBikes']} 台, "
                      f"Total Bike Capacity = {station['BikesCapacity']}, Bike Availability Rate = {station['BikeAvailabilityRate']:.2%}")
            
        time.sleep(600)
"""