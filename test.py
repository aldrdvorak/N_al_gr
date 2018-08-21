from netpy.nets import FeedForwardNet

net = FeedForwardNet(name = 'N_al_gr')

net.load_net_data()
net.load_weights()

print(net.forward(test))
