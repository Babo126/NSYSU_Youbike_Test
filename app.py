from flask import Flask, render_template
from bike_prediction import collect_data, predict_bike_availability_with_arima  # 導入 bike_prediction 中的函數
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    # 打開儲存的資料
    csv_path = "./data/國立中山大學幾何中心周圍1公里Youbike站點即時狀態.csv"
    data = pd.read_csv(csv_path)
    # 使用 ARIMA 模型預測自行車可用狀況
    low_bike_stations = predict_bike_availability_with_arima(data)

    # 過濾出可借車比例小於 20% 的站點，並計算還缺多少台車
    filtered_stations = []
    for station in low_bike_stations:
        # 計算車柱總數的 20%
        threshold = station['BikesCapacity'] * 0.2
        if station['PredictedAvailableBikes'] < threshold:
            # 還缺的車輛數量
            bike_shortage = int(threshold - station['PredictedAvailableBikes'])
            # 檢查 bike_shortage 是否大於 0，避免顯示還缺 0 台車的情況
            if bike_shortage > 0:
                filtered_stations.append({
                    'StationName': station['StationName'],
                    'BikeShortage': bike_shortage
                })

    # 獲取當前時間
    current_time = datetime.now().strftime('%m/%d %H:%M')

    # 將結果渲染到網頁模板上，並傳遞當前時間和更新後的數據
    return render_template('index.html', stations=filtered_stations, current_time=current_time)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # 從環境變量中獲取端口號，如果沒有則默認5000
    app.run(host='0.0.0.0', port=port)  # 監聽所有 IP 並在指定端口上運行
    