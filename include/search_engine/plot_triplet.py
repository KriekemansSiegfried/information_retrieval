from tensorflow_core.python.keras.utils.vis_utils import plot_model

from include.networks.network import get_network_triplet_loss, get_network

network = get_network_triplet_loss(19317, 2048, 512)
network = get_network(19317)
plot_model(network, to_file='model_simple.png', show_shapes=True, show_layer_names=True)
