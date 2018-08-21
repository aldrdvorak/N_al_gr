from netpy.teachers import BackPropTeacher_grad
from netpy.nets import FeedForwardNet
import numpy as np
from PIL import Image
import random
import math
from netpy.tools.errors import mse_Error
from netpy.tools.functions import iq_test
import h5py
import os
import copy
import datetime
from for_train_AVD import trainer_AVD



net = FeedForwardNet(name='N_al_gr')

f = trainer_AVD(net)

net.load_net_data()
net.load_weights()

#net.load_gradient()
f.read_gradient('N_al_gr_data/GRAD_W.h5')

l_r = 0.01
l_r_old = 0.1

step_w=10

teacher = BackPropTeacher_grad(net,error = 'MSE',learning_rate = [l_r, l_r_old])
teacher.learning_rate = [l_r, l_r_old]

path_Top = 'N_al_gr_data\IQ_top'
W_array = f.append_matrix(path_Top)

net.norm_G = 0

for i_epoch in range(1000):
    f.X, f.Y, f.name_of_answer_array = f.create_data_set(35)

    N_min, N_max, W_array = f.Calc_IQ_min_max(W_array)
    f.From_W_arr_to_Net(W_array[N_max])

    step_w = 0
    for i_w in range(len(W_array)):
        step_w+=W_array[i_w]['step_max']
    step_w/= len(W_array)

    f.print_w_arr(W_array,'------- Start epoch ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))

    nn, W_array[N_min], r_1 = f.Find_max_on_Line(W_array[N_min],W_array[N_max])
    N_min, N_max, W_array = f.Calc_IQ_min_max(W_array)

    W_array[N_max] = f.write_top_w_al(W_array[N_max])
    Rec_grad = f.calc_w_Grad(W_array,N_min,N_max)
    f.dd_r_Grad_to_Net(Rec_grad, 0.1)

    f.print_w_arr(W_array,'------- Start epoch_2 ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))

    W_array = f.Normalize(W_array, N_min, N_max)

    N_min, N_max, W_array = f.Calc_IQ_min_max(W_array)
    Rec_grad = f.calc_w_Grad(W_array,N_min,N_max)
    f.Add_r_Grad_to_Net(Rec_grad, 0.7)

    f.print_w_arr(W_array,'------- Start epoch_3 ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))
    f.From_W_arr_to_Net(W_array[N_max])


    for ii_epoch in range(2):
        net.save_gradient('N_al_gr_data/GRAD_W.h5')
        f.print_w_arr(W_array,' i = ' +str(i_epoch) +
                            ' ii=' + str(ii_epoch)+
                            ' ---- n_GR = ' +
                            '{0:.2f}'.format(net.norm_G)+
                            ' -----Learn_rate = ' + '{0:.4f}'.format(l_r))
        N_min, N_max    = f.find_i_min_max_in_iq(W_array, N_min, N_max )
        f.print_w_arr(W_array,' i = ' +str(i_epoch) +
                            ' ii=' + str(ii_epoch)+
                            ' ---2 n_GR = ' +
                            '{0:.2f}'.format(net.norm_G)+
                            ' -----Learn_rate = ' + '{0:.4f}'.format(l_r))

        IQ_in = W_array[N_max]['iq']

        f.X, f.Y, f.name_of_answer_array = f.sorted_Test_data(f.X, f.Y, f.name_of_answer_array)


        #con = input('Press any key...')

        net.iq, net.iq_min, net.iq_max =iq_test(f.net, f.X, f.Y)



        teacher.train(f.X, f.Y, 25, #learning_rate = [l_r, l_r_old],
                                 name_of_answer = name_of_answer_array,
                                 random_data = True,
                                 save_output_data = True)

        net.iq, net.iq_min, net.iq_max = f.iq_test(net, X, Y)

        R_train = f.Net_to_W_arr(W_array[N_max],'tr_'
                                + str(net.total_epoch))


        W_array.append(R_train)
        print(i_epoch, ii_epoch, " ---- End train -- " , R_train['name'],
                    " IQ= ", '{0:.4f}'.format(net.iq),
                    ' +- ','{0:.6f}'.format(net.iq -net.iq_min))

        N_min, N_max, W_array = f.Calc_IQ_min_max(W_array)

        IQ_out = W_array[N_max]['iq']
        f.print_w_arr(W_array,'-- End epoch-- i= ' +str(i_epoch) +
                    ' ii=' + str(ii_epoch) +
                    '--norm_G = ' + '{0:.4f}'.format(net.norm_G) +
                    '------ Delta IQ = '+ '{0:.6f}'.format(IQ_out - IQ_in)
                    )
