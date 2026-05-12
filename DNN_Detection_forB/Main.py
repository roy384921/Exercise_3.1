import os
import numpy as np
import matplotlib.pyplot as plt
from Train import train
from Test import test

# 定義系統配置類別
class sysconfig(object):
    def __init__(self, pilots=8, snr=20):
        self.Pilots = pilots              # 導頻數量
        self.with_CP_flag = True          # 是否包含循環前綴
        self.SNR = snr                    # 信噪比
        self.Clipping = False
        self.Train_set_path = '../H_dataset/'
        self.Test_set_path = '../H_dataset/'
        self.Model_path = f'../Models/Pilot_{pilots}_SNR_{snr}/'
        # 預設處理 16 個位元 (對應 8 個 DNN 中的一個，QPSK 模式)
        self.pred_range = np.arange(16, 32) 
        self.learning_rate = 0.001
        self.learning_rate_decrease_step = 2000

def run_simulations():
    # 依照題目要求設定測試參數
    snr_list = [5, 10, 15, 20, 25]
    pilot_list = [64, 16, 8, 0] # 包含 no-pilot 情況
    
    results = {} # 用於儲存 BER 數據

    for p in pilot_list:
        ber_at_snr = []
        for snr in snr_list:
            print(f"\n>>> 正在執行: Pilots={p}, SNR={snr}dB")
            config = sysconfig(pilots=p, snr=snr)
            
            # 確保模型儲存路徑存在
            if not os.path.exists(config.Model_path):
                os.makedirs(config.Model_path)

            # 1. 訓練模型
            train(config)
            
            # 2. 測試模型並取得 BER
            # 註：假設 Test.py 中的 test() 函數會回傳該次的 BER 數值
            current_ber = test(config) 
            ber_at_snr.append(current_ber)
            
        results[p] = ber_at_snr

    # 3. 繪製 SNR vs BER 圖表
    plot_results(snr_list, results)

def plot_results(snr_list, results):
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'x']
    
    for i, (p, ber_values) in enumerate(results.items()):
        label = f'Pilots: {p}' if p > 0 else 'No Pilot'
        plt.semilogy(snr_list, ber_values, marker=markers[i], label=label)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('FC-DNN Signal Detection Performance (Task b)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('BER_Analysis_Figure_3_3.png')
    print("\n>>> 圖表已儲存為 BER_Analysis_Figure_3_3.png")
    plt.show()

if __name__ == '__main__':
    run_simulations()

