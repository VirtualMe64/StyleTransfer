import torch
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io import read_image

import matplotlib.pyplot as plt

def load_image(path):
    return read_image(path).float()

def display_image(image_tensor : torch.Tensor):
    plt.imshow(image_tensor.int().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def generate_noisy_image(shape):
    im = torch.randint(0, 256, shape).float()
    return im

class StyleTranferModel(torch.nn.Module):
    def __init__(self):
        super(StyleTranferModel, self).__init__()

        # load model and set to eval mode
        model = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features.eval()
        return_nodes = {
            '4': 'conv1_1',
            '9': 'conv2_1',
            '16': 'conv3_1',
            '23': 'conv4_1',
            '30': 'conv5_1'
        }
        self.vgg_features_model = create_feature_extractor(model, return_nodes=return_nodes)

        # freeze params to improve performances   
        for param in self.vgg_features_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = self.vgg_features_model(x)
        return features

def content_loss(output, target, layer='conv1_1'):
    return 0.5 * torch.mean((output[layer] - target[layer]) ** 2)

def transfer_style(photo_path, iters = 250):
    # Load model and images
    m = StyleTranferModel()
    target_im = load_image(photo_path)
    target_features = m.forward(target_im)

    # Create noisy image which will become style transferred photo
    image = generate_noisy_image(target_im.shape)
    image.requires_grad = True

    # Create optimizer
    optimizer = torch.optim.Adam([image], lr = 1)

    for i in range(iters):
        optimizer.zero_grad() # reset grads
        
        # forward pass
        output = m.forward(image)
        loss = content_loss(output, target_features)
        
        # backward pass
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.data.clamp_(0, 255)

        if i % 50 == 0:
            print(f"Iter: {i}, Loss: {loss}")
            display_image(image.detach())

    display_image(image)

if __name__ == "__main__":
    transfer_style("photos/neckarfront.jpg", iters = 1000)