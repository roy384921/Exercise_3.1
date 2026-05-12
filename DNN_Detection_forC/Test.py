import numpy as np

# [修復 TensorFlow 相容性]
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import math
import os
from utils import *

# 加入 config 參數以承接 main.py 傳遞的值
def test(config):        
        tf.reset_default_graph()
        K = 64
        mu = 6
        P = config.Pilots
        SNRdb = config.SNR
        payloadBits_per_OFDM = K * mu
        H_folder = config.Test_set_path
        CP = K//4
        CP_flag = config.with_CP_flag
        Clipping_Flag = config.Clipping

        allCarriers = np.arange(K)
        if P<K:
            pilotCarriers = allCarriers[::K//P] 
            dataCarriers = np.delete(allCarriers, pilotCarriers)
        else:   
            pilotCarriers = allCarriers
            dataCarriers = []
            
        Pilot_file_name = 'Pilot_'+str(P)+'.txt'
        if os.path.isfile(Pilot_file_name):
            bits_pilot = np.loadtxt(Pilot_file_name, delimiter=',')
        else:
            bits_pilot = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        pilotValue = Modulation(bits_pilot, mu)

        # Network Parameters
        n_hidden_1 = 500
        n_hidden_2 = 250 
        n_hidden_3 = 120 
        n_input = 256 
        n_output = 48

        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_output])
        
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),            
            }
            biases = {            
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),          
            }
        
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
            return layer_4

        y_pred = encoder(X)
        y_true = Y

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        test_idx_low = 301
        test_idx_high = 401      

        channel_response_set_test = []
        for test_idx in range(test_idx_low,test_idx_high):
            H_file = H_folder + str(test_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    numbers_str = line.split()
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                    channel_response_set_test.append(h_response)

        print('length of testing channel response', len(channel_response_set_test))

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True

        with tf.Session(config=config_gpu) as sess:
            sess.run(init)            
            
            # 使用最新儲存的模型權重
            saving_name = config.Model_path + 'DetectionModel_SNR_' + str(SNRdb) + '_Pilot_' + str(P)
            saver.restore(sess, saving_name)            
            
            input_samples_test = []
            input_labels_test = []
            test_number = 500      
            
            for i in range(0, test_number):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                
                signal_output, para = ofdm_simulate(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
                
                input_labels_test.append(bits[config.pred_range])
                input_samples_test.append(signal_output)
                        
            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))                
            BER_tensor = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
                        
            final_ber = BER_tensor.eval({X:batch_x})
            print("OFDM Detection QAM output number is", n_output, "SNR = ", SNRdb, "Num Pilot", P, "BER on test set:", final_ber)
            
            # 回傳給 main.py 畫圖用
            return final_ber

