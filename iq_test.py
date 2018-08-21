from netpy.nets import FeedForwardNet
from netpy.tools.functions import iq_test
import numpy as np

net = FeedForwardNet(name = 'N_al_gr')

net.load_net_data()
net.load_weights()

test_X = np.loadtxt('data_set/test_X.txt')
test_Y = np.loadtxt('data_set/test_Y.txt')

print(iq_test(net, test_X, test_Y))
