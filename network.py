''' It's network init file '''

from netpy.nets import FeedForwardNet
from netpy.modules import LinearLayer, SigmoidLayer, FullConnection

net = FeedForwardNet(name = 'N_al_gr')


# Add your layers here
input_layer = LinearLayer(1024)
hidden_layer = SigmoidLayer(100)
hidden_layer2 = SigmoidLayer(100)
output_layer = SigmoidLayer(1024)

# Add your layers to the net here
net.add_Input_Layer(input_layer)
net.add_Layer(hidden_layer)
net.add_Layer(hidden_layer2)
net.add_Output_Layer(output_layer)

# Add your connections here
con_in_hid = FullConnection(input_layer, hidden_layer)
con_hid_hid2 = FullConnection(hidden_layer, hidden_layer2)
con_hid2_out = FullConnection(hidden_layer2, output_layer)

# Add your connections to the net here
net.add_Connection(con_in_hid)
net.add_Connection(con_hid_hid2)
net.add_Connection(con_hid2_out)

# Save your net
net.save()
