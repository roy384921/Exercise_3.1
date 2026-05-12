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
        self.Train_set_path = '../H_dataset/' 
        self.Test_set_path = '../H_dataset/'
        
        self.Model_path = f'../Models/SingleDNN_Pilot_{pilots}_SNR_{snr}/'
        
        self.pred_range = np.arange(0, 128) 
        
        self.learning_rate = 0.001
        self.learning_rate_decrease_step = 2000

def run_simulations():
    snr_list = [5, 10, 15, 20, 25]
    p = 8
    
    ber_single_dnn = []
    
    for snr in snr_list:
        print(f"\n========== [Task d: Single Large DNN] Pilots={p}, SNR={snr}dB ==========")
        config = sysconfig(pilots=p, snr=snr)
        if not os.path.exists(config.Model_path):
            os.makedirs(config.Model_path)

        train(config)
        current_ber = test(config) 
        ber_single_dnn.append(current_ber)

    # 繪製圖表
    plt.figure(figsize=(9, 6))
    plt.semilogy(snr_list, ber_single_dnn, marker='o', color='red', linewidth=2, label='Single Large DNN (128 outputs)')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Task (d): Single Large DNN vs 8 Small DNNs (QPSK, Pilots=8)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('BER_Analysis_Task_d.png')
    print("\n>>> 圖表已儲存為 BER_Analysis_Task_d.png")
    plt.show()

if __name__ == '__main__':
    run_simulations()

