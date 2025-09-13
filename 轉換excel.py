import pandas as pd
import os #連接電腦作業系統，取得修改，刪除檔案的權限

# 定義文件路徑
txt_file = 'output_data.txt'  # 原始檔案文件名稱
output_file = 'output_data.xlsx'  # 輸出的 Excel 文件

# 如果文件已存在，進行刪除的動作
if os.path.exists(output_file):
    os.remove(output_file)

# 定義處理函數
def process_line(line):
    try:
        # 假設數據格式為：Timestamp, Signal: 500 | BPM: 72
        if ',' in line and '|' in line:
            parts = line.split(',')  # 分割時間戳和信號部分
            timestamp = parts[0].strip()  # 提取時間戳

            # 提取 Signal 和 BPM
            signal = int(parts[1].split('|')[0].split(':')[1].strip())
            bpm = int(parts[1].split('|')[1].split(':')[1].strip())

            return timestamp, signal, bpm
        else:
            return None, None, None
    except Exception as e:
        print(f"處理行時發生錯誤：{line}，錯誤信息：{e}")
        return None, None, None

# 讀取和處理數據
data = []
with open(txt_file, 'r') as file:
    for line in file:
        line = line.strip()  # 移除首尾空格或換行符
        if line:
            timestamp, signal, bpm = process_line(line)
            if timestamp and signal and bpm:
                data.append({'Timestamp': timestamp, 'Signal': signal, 'BPM': bpm})

# 將數據轉為 DataFrame 並保存
if data:
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"數據已成功保存到 {output_file}")
else:
    print("沒有有效數據可保存")
