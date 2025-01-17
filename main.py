from torchvision.io import read_image
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

model = vgg16(weights = VGG16_Weights.IMAGENET1K_FEATURES).features

return_nodes = {
    '4': 'conv1_1',
    '9': 'conv2_1',
    '16': 'conv3_1',
    '23': 'conv4_1',
    '30': 'conv5_1'
}
model2 = create_feature_extractor(model, return_nodes=return_nodes)

def get_features(image_path):
    image = read_image(image_path).float()
    return model2(image)