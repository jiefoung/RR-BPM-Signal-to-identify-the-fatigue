import serial
import time

# 配置 Arduino 的串口名稱和波特率
SERIAL_PORT = 'COM3'  # 根據您的系統更改，例如 '/dev/ttyUSB0'（Linux/Mac）
BAUD_RATE = 115200      # 必須與 Arduino 中的 Serial.begin 設定一致
OUTPUT_FILE = 'output_data.txt'  # 儲存數據的 txt 檔案名稱

# 初始化串口通信
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"連接到 {SERIAL_PORT}，波特率：{BAUD_RATE}")
    ser.flush()  # 清空串口緩衝區
except Exception as e:
    print(f"無法打開串口 {SERIAL_PORT}，錯誤：{e}")
    exit()

# 打開文件並開始寫入數據
try:
    with open(OUTPUT_FILE, 'w') as file:
        file.write("Timestamp, Signal\n")  # 添加標題行

        while True:
            if ser.in_waiting > 0:  # 檢查是否有數據可讀
                data = ser.readline().decode('utf-8').strip()  # 讀取並解碼數據
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 獲取當前時間
                print(f"{timestamp}, {data}")  # 在終端輸出數據
                file.write(f"{timestamp}, {data}\n")  # 將數據寫入文件
                file.flush()  # 寫入硬碟
except KeyboardInterrupt:
    print("\n數據記錄已停止")
except Exception as e:
    print(f"發生錯誤：{e}")
finally:
    ser.close()  # 關閉串口
    print(f"數據已儲存到 {OUTPUT_FILE}")