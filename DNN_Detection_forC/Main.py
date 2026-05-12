import os
import numpy as np
import matplotlib.pyplot as plt
from Train import train
from Test import test

class sysconfig(object):
    def __init__(self, pilots=8, snr=20):
        self.Pilots = pilots              
        self.with_CP_flag = True          
        self.SNR = snr                    
        self.Clipping = False
        # 依照您先前的路徑修正，若只有一層 H_dataset 請改回 '../H_dataset/'
        self.Train_set_path = '../H_dataset/' 
        self.Test_set_path = '../H_dataset/'
        
        self.Model_path = f'../Models/64QAM_Pilot_{pilots}_SNR_{snr}/'
        
        # [Task c 重點修改]: 預測範圍改為 48 到 95 (總共 48 個 bit)
        self.pred_range = np.arange(48, 96) 
        
        self.learning_rate = 0.001
        self.learning_rate_decrease_step = 2000

def run_simulations():
    snr_list = [5, 10, 15, 20, 25]
    pilot_list = [64, 16, 8, 0] 
    results = {} 

    for p in pilot_list:
        ber_at_snr = []
        for snr in snr_list:
            print(f"\n========== [64-QAM] Pilots={p}, SNR={snr}dB ==========")
            config = sysconfig(pilots=p, snr=snr)
            if not os.path.exists(config.Model_path):
                os.makedirs(config.Model_path)

            train(config)
            current_ber = test(config) 
            ber_at_snr.append(current_ber)
            
        results[p] = ber_at_snr

    # 繪製圖表
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'x']
    for i, (p, ber_values) in enumerate(results.items()):
        label = f'Pilots: {p}' if p > 0 else 'No Pilot'
        plt.semilogy(snr_list, ber_values, marker=markers[i], label=label)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('FC-DNN Signal Detection (Task c: 64-QAM)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('BER_Analysis_Task_c_64QAM.png')
    print("\n>>> 圖表已儲存為 BER_Analysis_Task_c_64QAM.png")
    plt.show()

if __name__ == '__main__':
    run_simulations()
