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


def deform_pic(image, phi, coff_scale):
    new_image = np.zeros((32,32))

    for i in range(0, 32):
        for j in range(0, 32):
            new_image[i][j] = 255

    for i in range(0, 32):
        for j in range(0, 32):

            x0 = i - 15.5
            y0 = j - 15.5

            x1 = int(((x0*math.cos(phi) - y0*math.sin(phi))/coff_scale) + 15.5)
            y1 = int(((x0*math.sin(phi) + y0*math.cos(phi))/coff_scale) + 15.5)

            if x1 < 0:
                x1 = 0
            elif x1 > 31:
                x1 = 31

            if y1 < 0:
                y1 = 0
            elif y1 > 31:
                y1 = 0

            new_image[i][j] = image[x1][y1]

    image = new_image
    return image

def random_pic(num_canon):
    phi = random.uniform(-1, 1)
    scale = random.uniform(0.6, 1.4)

    canon = Image.open("canon/can_"+str(num_canon)+".png").convert('L')
    canon = np.array(canon)

    test = deform_pic(canon, phi, scale)

    canon = canon.flatten()/255
    test = test.flatten()/255

    return test, canon

def create_data_set(num_data):
    j = 0

    X = []
    Y = []
    name_of_answer_arr = []


    while j < num_data:
        for i in range(0, 36):
            test, canon = random_pic(i)

            X.append(test)
            Y.append(canon)
            name_of_answer_arr.append(i)

            j += 1

    return X, Y, name_of_answer_arr

def sorted_Test_data(X, Y, name_of_answer_array):

    errors_old = []
    errors_new = []
    nn=[]

    new_name_of_answer = []
    new_X = []
    new_Y = []

    for i in range(len(X)):
        errors_old = np.hstack((errors_old, mse_Error(net.forward(X[i]), Y[i])))
        nn.append(i)

    errors_new = np.hstack((errors_new, errors_old))

    n_up=1
    while n_up>0:
        n_up = 0
        for i in range(len(X)-1):
             if errors_new[i]<errors_new[i+1]:
                 err_i = errors_new[i]
                 nnn_i = nn[i]
                 errors_new[i] =errors_new[i+1]
                 nn[i] = nn[i+1]
                 errors_new[i+1]=err_i
                 nn[i+1] = nnn_i
                 n_up+=1

    for i in range(len(errors_new)):
            new_X.append((X[i]))
            new_Y.append((Y[i]))
            new_name_of_answer.append(name_of_answer_array[nn[i]])
            """
            new_X.append((X[nn[i]]))
            new_Y.append((Y[nn[i]]))
            new_name_of_answer.append(name_of_answer_array[nn[i]])
            """

    for i in range(40):
        print(i, "\t", name_of_answer_array[i],
                 "\t",'{0:.5f}'.format(errors_old[i]),
                 "\t", new_name_of_answer[i],
                 "\t",'{0:.5f}'.format(errors_new[i]))

    print('.......')

    for i in range(len(new_X)-35, len(new_X)):
        print(i, "\t", name_of_answer_array[i],
                 "\t",'{0:.5f}'.format(errors_old[i]),
                 "\t", new_name_of_answer[i],
                 "\t",'{0:.5f}'.format(errors_new[i]))

    return new_X, new_Y, new_name_of_answer

def append_matrix(path_):
    tmp_matrix = []
    ii=0
    for file in os.listdir(path_):
        if file.endswith('.h5'):
            ii+=1
            if ii>10:
                break
            tmp = {"name": None,"iq": None,"norm": None,"weights": []}
            tmp["name"] = file
            tmp["iq"] = 0
            tmp["iq_del"] = 0
            tmp["norm"] = 1
            tmp["step_max"] = 0
            matrix_arr = h5py.File(path_+'/'+file, 'r')
            for i in range(1, len(net.modules), 2):
                tmp["weights"].append(matrix_arr['weights_'+str(i-1)+'_'+str(i+1)][:])
            tmp_matrix.append(tmp)

    name_w = 'NE_' + str(net.total_epoch) + '_W.h5'
    tmp = Net_to_W_arr(tmp, name_w)
    tmp_matrix.append(tmp)

    return tmp_matrix

def read_gradient(filename):
    data = h5py.File(filename, 'r')
    summG = 0
    for i in range(1, len(net.modules), 2):
        current_matrix = data['gradient_'+str(i-1)+'_'+str(i+1)][:]
        for m in range(len(current_matrix)):
            for n in range(len(current_matrix[0])):
                net.modules[i].weight_gradient[m][n] = current_matrix[m][n]
                summG+=current_matrix[m][n]**2
    net.norm_G = math.sqrt(summG)
    data.close()

def From_W_arr_to_Net(Rec_W_arr):
    for i in range(1, len(net.modules), 2):
        j = int((i-1)/2)
        net.modules[i].weight_matrix = copy.deepcopy(Rec_W_arr["weights"][j])

def Net_to_W_arr(Rec_W_arr, name_new):
        R_new = copy.deepcopy(Rec_W_arr)
        R_new["name"] = name_new
        R_new["iq"] = net.iq
        for i in range(1, len(net.modules), 2):
            j = int((i-1)/2)
            for m in range(len(net.modules[i].weight_matrix)):
                for n in range(len(net.modules[i].weight_matrix[m])):
                    R_new["weights"][j][m][n]  = net.modules[i].weight_matrix[m][n]
        R_new["norm"] = calc_norm_rec
        return R_new

def Net_Grad_to_rec(Rec_W_arr):
        R_grad = copy.deepcopy(Rec_W_arr)
        R_grad["name"] = 'grad_Net'
        R_grad["iq"] = net.iq
        R_grad["norm"] = net.norm_G
        for i in range(1, len(net.modules), 2):
            j = int((i-1)/2)
            for m in range(len(net.modules[i].weight_gradient)):
                for n in range(len(net.modules[i].weight_gradient[m])):
                    R_grad["weights"][j][m][n]  = net.modules[i].weight_gradient[m][n]
        return R_grad

def Grad_to_Net(R_grad):
    net.norm_G = R_grad["norm"]
    for i in range(1, len(net.modules), 2):
        j = int((i-1)/2)
        net.modules[i].weight_gradient = copy.deepcopy(R_grad["weights"][j])

def calc_norm_rec(Rec_1):
      summG = 0
      for i in range(len(Rec_1["weights"])):
          for j in range(len(Rec_1["weights"][i])):
              for k in range(len(Rec_1["weights"][i][j])):
                  summG+= Rec_1["weights"][i][j][k] **2
      return math.sqrt(summG)

def Add_r_Grad_to_Net(R_grad, al):
    R_grad["norm"] = calc_norm_rec(R_grad)
    #koef_1 = 1 / (net.norm_G + 0.0001)
    koef_2 = al / ( R_grad["norm"]  + 0.01)
    if math.fabs(koef_2)<0.001:
            return
    s_n = 0
    s_c = 0
    for i in range(1, len(net.modules), 2):
        j = int((i-1)/2)
        for m in range(len(net.modules[i].weight_gradient)):
            for n in range(len(net.modules[i].weight_gradient[m])):
                #net.modules[i].weight_gradient[m][n]*= koef_1
                a_w = net.modules[i].weight_gradient[m][n]
                if a_w > -10000:
                    a_w+=0
                else:
                    a_w=0
                a_g = R_grad["weights"][j][m][n]
                if a_g > -10000:
                    a_g+=0
                else:
                    a_g =0

                s_c+= ( a_w * a_g)
                a_w+= (koef_2 * a_g)
                net.modules[i].weight_gradient[m][n] = a_w
                s_n+= a_w ** 2
                if s_n>0:
                    s_n+=0
                else:
                    s_n=0


    #fi_ = math.acos(fi_ / (2*al+0.00001))
    s_n = math.sqrt(s_n)
    if (s_n > 0):
        net.norm_G = s_n
    else:
        net.norm_G = 1
        print(s_n)
    fi_ =  s_c / (R_grad["norm"] * s_n)
    print('GRAD_NORM = '+ '{0:.4f}'.format(R_grad["iq"]),
                        ' al= '+ '{0:.5f}'.format(al),
                        ' NET_G_in= '+ '{0:.4f}'.format(net.norm_G),
                         '_out='+ '{0:.4f}'.format(s_n),
                        ' Cos_fi= '+ '{0:.2f}'.format(fi_)
                         )

def rec_test(rec_tmp):
    From_W_arr_to_Net(rec_tmp)
    rec_tmp['iq'], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    rec_tmp["norm"] = calc_norm_rec(rec_tmp)
    if rec_tmp["name"][0]=='M':
        rec_tmp["name"] = 'C' + rec_tmp["name"]

    return rec_tmp

# вычмисляет IQ для всех матриц из массива,
# записывает с наибоkьшим IQ в текущую сеть
def Calc_IQ_min_max(tmp_matrix):
    iq_min=100
    iq_max=0
    for i in range(len(tmp_matrix)):
        tmp_matrix[i] = rec_test(tmp_matrix[i])
        if tmp_matrix[i]['iq'] < iq_min:
                i_min = i
                iq_min = tmp_matrix[i]['iq']
        if tmp_matrix[i]['iq'] > iq_max:
                i_max = i
                iq_max = tmp_matrix[i]['iq']

    for i in range(len(tmp_matrix)):
        if i!=i_max:
            tmp_matrix[i]["iq_del"] = (tmp_matrix[i_max]['iq'] -
                                         tmp_matrix[i]['iq'])
            r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[i_max])
            tmp_matrix[i]["step_max"] = r1['norm']
        else:
            tmp_matrix[i]["step_max"] = 0
            tmp_matrix[i]["iq_del"] = 0

    tmp_matrix[i_max] = write_top_w_al(tmp_matrix[i_max])
    return i_min, i_max, tmp_matrix

def Normalize(tmp_matrix, N_min, N_max):
    #print('!!!!_1', len(tmp_matrix))
    for i in range(len(tmp_matrix)):
        if i!=N_max:
            r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[N_max])
            tmp_matrix[i]["step_max"] = r1['norm']
            if tmp_matrix[i]["step_max"] > 1:
                Add_r_Grad_to_Net(r1, 0.1 * (tmp_matrix[i]["iq_del"]
                                    /(tmp_matrix[i]["step_max"])))
                if tmp_matrix[i]["step_max"] > 100:
                    koef = (100/tmp_matrix[i]["step_max"])
                    tmp_matrix[i]= Move_Rec_Point_on_M(tmp_matrix[i],
                                r1, koef)
                    tmp_matrix[i]["name"] = 'ort_' + tmp_matrix[i]["name"]
            else:
                Rec_grad = Net_Grad_to_rec(r1)
                tmp_matrix[i]= Move_Rec_Point_on_M(tmp_matrix[i], Rec_grad,
                            100/Rec_grad['norm'])
                tmp_matrix[i] = rec_test(tmp_matrix[i])
                tmp_matrix[i]["name"] = 'grd_' + tmp_matrix[i]["name"]
        else:
            tmp_matrix[i]["step_max"] = 0
            tmp_matrix[i]["iq_del"] = 0
    i = 0
    while i<len(tmp_matrix):
        if i!=N_max:
            j = i+1
            while j<len(tmp_matrix):
                r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[j])
                if r1['norm']<1:
                    print(' between ' , i , tmp_matrix[i]["name"],
                            j,  tmp_matrix[j]["name"],
                            '{0:.2f}'.format(r1['norm'])
                            )
                    N_min, N_max, tmp_matrix = bonding_W(i,j,
                                        N_min, N_max,tmp_matrix)
                    print_w_arr(tmp_matrix, ' bonding i='  + str(i) + ' j=' +str(j))
                    break
                else:
                    if len(tmp_matrix[j]["name"])>30:
                        N_min, N_max, tmp_matrix = delete_W(j,
                                        N_min, N_max,tmp_matrix)
                j+=1
        i+=1
    while len(tmp_matrix)>7:
        N_min, N_max, tmp_matrix = delete_W(N_min, N_min, N_max,tmp_matrix)

    #print('!!!!_2', len(tmp_matrix))
    r1 = find_center_no_max(tmp_matrix, N_min, N_max)
    r1 = rec_test(r1)

    nn, r1, r_1_2 = Find_max_on_Line(r1, tmp_matrix[N_max])
    r1['name'] = 'C_' + str(net.total_epoch)
    tmp_matrix.append(r1)
    Add_r_Grad_to_Net(r_1_2, 0.7 *
            (tmp_matrix[N_max]["iq"] - r1["iq"]) / r_1_2["norm"])

    print('!!!!_3', len(tmp_matrix),
        'r1', r1['name'], r1['iq'],
        'norm', r_1_2["norm"],
        'w_max', tmp_matrix[N_max]['name'], tmp_matrix[N_max]["iq"])

    return tmp_matrix

def delete_W(n1, N_min, N_max,tmp_matrix):
    w_new=[]
    for i in range(len(tmp_matrix)):
        if (i!=n1):
            w_new.append(tmp_matrix[i])
    N_min, N_max = find_i_min_max_in_iq(w_new, N_min, N_max)
    return N_min, N_max, w_new

def bonding_W(n1,n2, N_min, N_max,tmp_matrix):
    w_new=[]
    for i in range(len(tmp_matrix)):
        if ((i!=n1) and (i!=n2)):
            w_new.append(tmp_matrix[i])
            r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[N_max])

    r_1 = Lin_W_on_al(0.5, tmp_matrix[n1], tmp_matrix[n2])
    r_1['name'] = 'b ' + tmp_matrix[n1]['name'] + '+' + tmp_matrix[n2]['name']
    w_new.append(r_1)
    #N_min, N_max, w_new = Calc_IQ_min_max(w_new)
    return N_min, N_max, w_new

def find_i_min_max_in_iq(tmp_matrix, N_min, N_max):
    iq_min=100
    iq_max=0
    N_m_old = N_max
    nn = len(tmp_matrix)
    for i in range(nn):
            iq_Test = tmp_matrix[i]["iq"]
            if iq_Test < iq_min:
                N_min = i
                iq_min = iq_Test
            if iq_Test > iq_max:
                N_max = i
                iq_max = iq_Test
    if  N_max!=N_m_old:
        for i in range(len(tmp_matrix)):
            if i!=N_max:
                tmp_matrix[i]["iq_del"] = (tmp_matrix[N_max]['iq'] -
                                         tmp_matrix[i]['iq'])
                r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[N_max])
                tmp_matrix[i]["step_max"] = r1['norm']
            else:
                tmp_matrix[i]["step_max"] = 0
                tmp_matrix[i]["iq_del"] = 0

    return   N_min, N_max

def find_center_no_max(tmp_matrix, N_min, N_max):
    r_center = copy.deepcopy(tmp_matrix[0])
    norm_G = 0
    kk = 1 / (len(tmp_matrix)-1)
    for i in range(len(r_center["weights"])):
        for j in range(len(r_center["weights"][i])):
            for k in range(len(r_center["weights"][i][j])):
                summG = 0
                for i_nn in range(nn):
                     if i_nn!=N_max:
                        summG+= tmp_matrix[i_nn]["weights"][i][j][k]
                summG*=kk
                r_center["weights"][i][j][k] = summG
                norm_G+= summG**2
    r_center['norm'] = math.sqrt(norm_G)
    r_center['name'] = 'C_' + str(net.total_epoch)
    return r_center


def Move_min_max_in_iq(tmp_matrix, step_min):
    for i in range(len(tmp_matrix)):
        if i!=i_max:
            tmp_matrix[i]["iq_del"] = (tmp_matrix[i_max]['iq'] -
                                     tmp_matrix[i]['iq'])
            r1 = Rec_from_to(tmp_matrix[i],tmp_matrix[i_max])
            tmp_matrix[i]["step_max"] = calc_norm_rec(r1)

            if tmp_matrix[i]["step_max"] < 0.1 *step_min:
                r1 = Net_Grad_to_rec(r1)
                tmp_matrix[i]  = Move_Rec_Point_on_M(tmp_matrix[i] ,
                                r1, step_min)
                tmp_matrix[i]["step_max"] = calc_norm_rec(
                        Rec_from_to(tmp_matrix[i],tmp_matrix[i_max]))
                tmp_matrix[i]['iq'], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
                tmp_matrix[i]["iq_del"] = (tmp_matrix[i_max]['iq'] -
                                        tmp_matrix[i]['iq'])

            if tmp_matrix[i]["step_max"] < step_min:
                tmp_matrix[i]  = Move_Rec_Point_on_M(tmp_matrix[i] ,
                                r1 , 3 * tmp_matrix[i]["step_max"] )
                tmp_matrix[i]["step_max"] = calc_norm_rec(
                        Rec_from_to(tmp_matrix[i],tmp_matrix[i_max]))
                tmp_matrix[i]['iq'], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
                tmp_matrix[i]["iq_del"] = (tmp_matrix[i_max]['iq'] -
                                        tmp_matrix[i]['iq'])

            if tmp_matrix[i]['iq'] < iq_min:
                            i_min = i
                            iq_min = tmp_matrix[i]['iq']
            if  tmp_matrix[i]["step_max"] >0:
                    i = i
            else:
                    tmp_matrix[i]["step_max"] = 1000
                    tmp_matrix[i]['iq'] = tmp_matrix[i_min]['iq']
                    tmp_matrix[i]["iq_del"] = tmp_matrix[i_min]['iq_del']

        else:
            tmp_matrix[i]["step_max"] = 0
            tmp_matrix[i]["iq_del"] = 0


    return   i_min, i_max

def print_w_arr(W_array, txt_):
        now = datetime.datetime.now()
        print(now.strftime("%d-%m-%Y  %H:%M-%S"), txt_ ," ------------- IQ= ",
                    '{0:.5f}'.format(net.iq), ' +- ',
                    '{0:.5f}'.format(net.iq -net.iq_min))
        for i in range(len(W_array)):
            strM=''
            if i==N_min:
                strM='  <--- Min'
            if i==N_max:
                strM='  <--- Max'
            print(' i = ', i ,
                    ' step_max = ','{0:.1f}'.format(W_array[i]["step_max"]),
                     "\t", ' IQ = ','{0:.4f}'.format(W_array[i]["iq"]),
                    ' iq_del = ','{0:.4f}'.format(W_array[i]["iq_del"]),
                      "\t", W_array[i]["name"],
                    strM)

def write_top_w_al(Rec_W):
    name_W = Rec_W['name']
    if name_W[0]!='N':
        net.total_epoch+=1
        name_W_new = 'NE_' + str(net.total_epoch) + '_W.h5'
        path_W_top = 'N_al_gr_data/IQ_top/' + name_W_new
        net.save_weights(path_W_top)
        print(' ------------------- New top weights!  ----------------- IQ= ',
                    '{0:.4f}'.format(Rec_W['iq']) + ' -----------: ',
                    name_W + '---->' + name_W_new)
        Rec_W['name']= name_W_new
    return Rec_W



def calc_w_Grad(W_arr, N_min, N_max):
    Rec_Grad = copy.deepcopy(W_arr[N_max])
    nn = len(W_arr)
    K_g = []
    norm_G = 0
    #Rec_new = Rec_middle
    for i_nn in range(nn):
        if i_nn!=N_max:
            K_g.append((W_arr[N_max]['iq']-W_arr[i_nn]['iq'])/(
                        W_arr[i_nn]['step_max'] + 0.000001))
        else:
            K_g.append(0)

    for i in range(len(Rec_Grad["weights"])):
        for j in range(len(Rec_Grad["weights"][i])):
            for k in range(len(Rec_Grad["weights"][i][j])):
                summG = 0
                for i_nn in range(nn):
                     if i_nn!=N_max:
                        summG+= K_g[i_nn] * (
                                W_arr[N_max]["weights"][i][j][k]-
                                W_arr[i_nn]["weights"][i][j][k])
                Rec_Grad["weights"][i][j][k] = summG
                norm_G+= summG**2
    norm_G = math.sqrt(norm_G)
    Rec_Grad["name"] = 'G_' + str(net.total_epoch)
    Rec_Grad["norm"] = norm_G
    Rec_Grad["iq"] = 1
    return Rec_Grad

def calc_summ_ort_norm(Rec_1, Rec_2):
      Rec_S = copy.deepcopy(Rec_1)
      #Rec_new = Rec_middle
      Rec_1['norm'] = calc_norm_rec(Rec_1)
      Rec_2['norm'] = calc_norm_rec(Rec_2)
      n_1 = 1 / Rec_1['norm']
      n_2 = 1 / Rec_2['norm']
      k_1_2 = math.sqrt(Rec_1['norm']*Rec_1['norm'])
      summG = 0
      for i in range(len(Rec_1["weights"])):
          for j in range(len(Rec_1["weights"][i])):
              for k in range(len(Rec_1["weights"][i][j])):
                  aa = n_1*Rec_S["weights"][i][j][k]
                  aa+= n_2*Rec_2["weights"][i][j][k]
                  aa*= k_1_2
                  summG+= aa**2
                  Rec_S["weights"][i][j][k] = aa
      Rec_S['norm'] = math.sqrt(summG)
      Rec_S["name"] = Rec_1["name"] + '+N+' + Rec_2["name"]

      return Rec_S

def Mult_Scalar_rec(Rec_1, Rec_2):
    summG = 0
    for i in range(len(Rec_1["weights"])):
        for j in range(len(Rec_1["weights"][i])):
            for k in range(len(Rec_1["weights"][i][j])):
                summG+= (Rec_1["weights"][i][j][k] *Rec_2["weights"][i][j][k])
    return summG

def Find_max_on_Line(r_1, r_2):
    delta_al=0.5
    almax_old=0
    delta_s = 1000
    nn=0

    if r_1['iq'] < r_2['iq']:
        r_min = r_1
        r_max = copy.deepcopy(r_2)
    else:
        r_min = r_2
        r_max = copy.deepcopy(r_1)

    while ((delta_al>0.01) and (nn<10) and delta_s>1):
        nn+=1
        r_1_2 = Rec_from_to(r_min,r_max)
        delta_s = r_1_2['norm']
        almax , r_3 = Find_al_max_on_05(r_min, r_max)
        delta_al = math.fabs(r_3['iq'] - r_max['iq'])
        iq_min = r_min['iq']
        iq_max = r_max['iq']
        iq_new = r_3['iq']
        #print('{0:.5f}'.format(r_max['iq']),'{0:.5f}'.format(r_3['iq']), '{0:.5f}'.format(delta_al))

        if (r_3['iq']>r_max['iq']):
            r_min = r_max
            r_max = copy.deepcopy(r_3)
            str_='var_Best'
        else:
            str_='var_0.95'
            almax = 0.95
            r_3 = Lin_W_on_al(0.95, r_min, r_max)
            From_W_arr_to_Net(r_3)
            r_3["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
            iq_new = r_3["iq"]
            delta_al = math.fabs(r_3['iq'] - r_max['iq'])
            if (r_3['iq']>r_max['iq']):
                    str_='yes_0.95'
                    r_min = r_max
                    r_max = copy.deepcopy(r_3)
            else:
                    r_min = r_3
                    str_='var_s_so'
        print('   Find_on_Line  n=', nn, str_ ,  "\t",
                '{0:.4f}'.format(iq_min), '{0:.4f}'.format(iq_max),
                '{0:.4f}'.format(iq_new),
                '<--al=', '{0:.2f}'.format(almax),
                ' delta=', '{0:.4f}'.format(delta_al),
                ' del_s=', '{0:.1f}'.format(delta_s),
                )
    r_max['name'] = 'FL_'+str(net.total_epoch)
    r_1_2 = Rec_from_to(r_min,r_max)
    #From_W_arr_to_Net(r_max)
    return nn, r_max, r_1_2

def Find_al_max_on_05(r_0, r_1):
    al_max= 0.5
    r_2 = Lin_W_on_al(al_max, r_0, r_1)
    From_W_arr_to_Net(r_2)
    r_2["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    i_05 = r_2["iq"]
    aa =2* (2 * r_1["iq"]+ 2 * r_0["iq"] - 4* i_05)
    if aa<0:
        al_max= (r_1["iq"] + 3* r_0["iq"] - 4*i_05) / aa
        r_2 = Lin_W_on_al(al_max, r_0, r_1)
        From_W_arr_to_Net(r_2)
        r_2["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    return al_max, r_2

def Normal_Component_on_G(Rec_1, Rec_G):
    Rec_N = copy.deepcopy(Rec_1)
    scalar = Mult_Scalar_rec(Rec_1, Rec_G)
    Koef = scalar / (Rec_G['norm'])
    for i in range(len(Rec_1["weights"])):
        for j in range(len(Rec_1["weights"][i])):
            for k in range(len(Rec_1["weights"][i][j])):
                Rec_N["weights"][i][j][k]-= Koef * Rec_G["weights"][i][j][k]

    Rec_N["iq"] = scalar
    Rec_N["name"] = 'P_norm' + str(net.total_epoch)
    Rec_N["norm"] = calc_norm_rec(Rec_N)
    return scalar, Koef, Rec_N

def Move_Rec_Point_on_M(Rec_P, Rec_M, KoefStep):
    R_1 = copy.deepcopy(Rec_P)
    for i in range(len(Rec_P["weights"])):
        for j in range(len(Rec_P["weights"][i])):
            for k in range(len(Rec_P["weights"][i][j])):
                aa = KoefStep*Rec_M["weights"][i][j][k]
                R_1["weights"][i][j][k]+= aa
    From_W_arr_to_Net(R_1)
    R_1["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    R_1["name"] = Rec_P["name"]
    if R_1["name"][0]!='M':
        R_1["name"] = 'M_' + R_1["name"]
    return R_1

def Lin_W_on_al(alpha, r_From, r_To):
    R_new =  copy.deepcopy(r_From)
    R_new["name"] = 'al=' + '{0:.3f}'.format(alpha)
    R_new["iq"] = 0
    for ii in range(len(r_From["weights"])):
        for m in range(len(r_From["weights"][ii])):
            for n in range(len(r_From["weights"][ii][m])):
                R_new["weights"][ii][m][n]= ((1- alpha) *
                                        r_From["weights"][ii][m][n] +
                                        alpha * r_To["weights"][ii][m][n])
    R_new["norm"] = calc_norm_rec(R_new)
    return R_new

def Rec_from_to(r_From, r_To):
    R_new =  copy.deepcopy(r_To)
    R_new["name"] = r_From["name"]+'->' + r_To["name"]
    s_r= 0
    for ii in range(len(r_From["weights"])):
        for m in range(len(r_From["weights"][ii])):
            for n in range(len(r_From["weights"][ii][m])):
                R_new["weights"][ii][m][n]-= r_From["weights"][ii][m][n]
                s_r+= R_new["weights"][ii][m][n]**2
    R_new['norm'] = math.sqrt(s_r)
    return R_new

def Find_max_on_2(r_0, r_1):
    r_2 = Lin_W_on_al(2, r_0, r_1)
    From_W_arr_to_Net(r_2)
    r_2["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    iq_2=r_2["iq"]

    a = (r_0["iq"] +r_2["iq"] - 2* r_1["iq"]) / 2
    b = r_1["iq"] - r_0["iq"] - a
    al_max = - b / (2*a)
    extr = a*  al_max *al_max + b * al_max  + r_0["iq"]


    if ((r_2["iq"] > r_1["iq"]) and  (r_2["iq"] > r_0["iq"]) and
                                (extr < r_2["iq"])):
        al_max = 2
        print(' ', r_2["name"] ,'->','{0:.4f}'.format(r_0["iq"]),
                                            '{0:.4f}'.format(r_1["iq"]),
                                            '{0:.4f}'.format(iq_2),
                                            '---------->',
                                            '{0:.4f}'.format(r_2["iq"]))
        return r_2

    if ((extr < r_1["iq"]) and  (extr < r_0["iq"])):
        if (r_0["iq"] < r_1["iq"]):
            al_max = 1.1
        else:
            al_max = -0.1

    if al_max>15:
        al_max=15
    if al_max<-2:
        al_max=-2
    al_max

    r_2 = Lin_W_on_al(al_max, r_0, r_1)
    From_W_arr_to_Net(r_2)
    r_2["iq"], iq_Test_min, iq_Test_max =iq_test(net, X, Y)
    print('   M_al :', r_2["name"] ,'EXTR_2=','{0:.3f}'.format(extr),'->',
                                            '{0:.3f}'.format(r_0["iq"]),
                                            '{0:.3f}'.format(r_1["iq"]),
                                            '{0:.3f}'.format(iq_2),
                                            '---------->',
                                            '{0:.3f}'.format(r_2["iq"]))
    return r_2



net = FeedForwardNet(name='N_al_gr')


net.load_net_data()
net.load_weights()

#net.load_gradient()
read_gradient('N_al_gr_data/GRAD_W.h5')

l_r = 0.01
l_r_old = 0.1

step_w=10

teacher = BackPropTeacher_grad(net,error = 'MSE',learning_rate = [l_r, l_r_old])
teacher.learning_rate = [l_r, l_r_old]

path_Top = 'N_al_gr_data\IQ_top'
W_array = append_matrix(path_Top)

net.norm_G = 0

for i_epoch in range(1000):
    X, Y, name_of_answer_array = create_data_set(35)

    N_min, N_max, W_array = Calc_IQ_min_max(W_array)
    From_W_arr_to_Net(W_array[N_max])

    step_w = 0
    for i_w in range(len(W_array)):
        step_w+=W_array[i_w]['step_max']
    step_w/= len(W_array)

    print_w_arr(W_array,'------- Start epoch ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))

    nn, W_array[N_min], r_1 = Find_max_on_Line(W_array[N_min],W_array[N_max])
    N_min, N_max, W_array = Calc_IQ_min_max(W_array)

    W_array[N_max] = write_top_w_al(W_array[N_max])
    Rec_grad = calc_w_Grad(W_array,N_min,N_max)
    Add_r_Grad_to_Net(Rec_grad, 0.1)

    print_w_arr(W_array,'------- Start epoch_2 ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))

    W_array = Normalize(W_array, N_min, N_max)

    N_min, N_max, W_array = Calc_IQ_min_max(W_array)
    Rec_grad = calc_w_Grad(W_array,N_min,N_max)
    Add_r_Grad_to_Net(Rec_grad, 0.7)

    print_w_arr(W_array,'------- Start epoch_3 ' + str(i_epoch) +  ' ii=0 ' +
                        '------- Step =  ' + '{0:.5f}'.format(step_w))
    From_W_arr_to_Net(W_array[N_max])


    for ii_epoch in range(2):
        net.save_gradient('N_al_gr_data/GRAD_W.h5')
        print_w_arr(W_array,' i = ' +str(i_epoch) +
                            ' ii=' + str(ii_epoch)+
                            ' ---- n_GR = ' +
                            '{0:.2f}'.format(net.norm_G)+
                            ' -----Learn_rate = ' + '{0:.4f}'.format(l_r))
        N_min, N_max    = find_i_min_max_in_iq(W_array, N_min, N_max )
        print_w_arr(W_array,' i = ' +str(i_epoch) +
                            ' ii=' + str(ii_epoch)+
                            ' ---2 n_GR = ' +
                            '{0:.2f}'.format(net.norm_G)+
                            ' -----Learn_rate = ' + '{0:.4f}'.format(l_r))

        IQ_in = W_array[N_max]['iq']

        #X, Y, name_of_answer_array = sorted_Test_data(X, Y, name_of_answer_array)


        #con = input('Press any key...')

        net.iq, net.iq_min, net.iq_max =iq_test(net, X, Y)



        teacher.train(X, Y, 25, #learning_rate = [l_r, l_r_old],
                                 name_of_answer = name_of_answer_array,
                                 random_data = True,
                                 save_output_data = True)

        net.iq, net.iq_min, net.iq_max =iq_test(net, X, Y)

        R_train = Net_to_W_arr(W_array[N_max],'tr_'
                                + str(net.total_epoch))


        W_array.append(R_train)
        print(i_epoch, ii_epoch, " ---- End train -- " , R_train['name'],
                    " IQ= ", '{0:.4f}'.format(net.iq),
                    ' +- ','{0:.6f}'.format(net.iq -net.iq_min))

        """
        r_Max_TR = Rec_from_to(W_array[N_max],R_train)
        Add_r_Grad_to_Net(r_Max_TR, 0.1 * ((net.iq - W_array[N_max]['iq'])
                                    /(r_Max_TR['norm']+0.0001)))
        print(i_epoch, ii_epoch, " ---- End train -- " , R_train['name'],
                    " IQ= ", '{0:.4f}'.format(net.iq),
                    ' +- ','{0:.6f}'.format(net.iq -net.iq_min))


        N_old = N_max
        nn, W_array[N_min], r_1 = Find_max_on_Line(W_array[N_min],W_array[N_max])
        N_min, N_max    = find_i_min_max_in_iq(W_array, N_min, N_max )
        if N_max!=N_old:
            Rec_grad = Net_Grad_to_rec(Rec_grad)
            W_array[N_old] = Move_Rec_Point_on_M(W_array[N_old], Rec_grad, 0.1 )

        From_W_arr_to_Net(R_train)
        W_array[N_min]  = R_train
        N_min, N_max    = find_i_min_max_in_iq(W_array, N_min, N_max )

        print_w_arr(W_array,'  -After move w_min--->w_max- ')


        """
        N_min, N_max, W_array = Calc_IQ_min_max(W_array)

        IQ_out = W_array[N_max]['iq']
        print_w_arr(W_array,'-- End epoch-- i= ' +str(i_epoch) +
                    ' ii=' + str(ii_epoch) +
                    '--norm_G = ' + '{0:.4f}'.format(net.norm_G) +
                    '------ Delta IQ = '+ '{0:.6f}'.format(IQ_out - IQ_in)
                    )
