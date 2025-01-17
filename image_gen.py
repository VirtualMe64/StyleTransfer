import torch
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io import read_image
from torchvision.utils import save_image

import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "output"

def load_image(path):
    return read_image(path).float() / 255

def display_image(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def checkpoint(image_tensor, photo_path, painting_path, alpha, beta, iteration):
    photo_name = os.path.splitext(os.path.basename(photo_path))[0]
    painting_name = os.path.splitext(os.path.basename(painting_path))[0]

    subfolder = f"{photo_name}-{painting_name}-a{alpha}_b{beta}"
    folder = os.path.join(OUTPUT_FOLDER, subfolder)
    path = os.path.join(folder, f"{iteration:05}.jpg")

    if not os.path.exists(folder):
        os.makedirs(folder)
    save_image(image_tensor, path)

def generate_noisy_image(shape):
    im = torch.rand(shape).float()
    return im

def compute_gram_matrix(features : torch.Tensor):
    return torch.einsum('ijk,ljk->il', features, features)

class StyleTranferModel(torch.nn.Module):
    def __init__(self):
        super(StyleTranferModel, self).__init__()

        # load model and set to eval mode
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
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

def content_loss(output, target, layer='conv3_1'):
    return 0.5 * torch.sum((output[layer] - target[layer]) ** 2)

def style_loss(output, target_grams):
    total_loss = 0
    for layer in output.keys():
        output_gram = compute_gram_matrix(output[layer])
        err = torch.sum((output_gram - target_grams[layer]) ** 2)
        err /= 4
        err /= output[layer].shape[0] ** 2
        err /= (output[layer].shape[1] * output[layer].shape[2]) ** 2
        total_loss += 0.2 * err
    return total_loss

def transfer_style(photo_path, painting_path, alpha = 1, beta = 10, epoch_size = 100, iters = 20):
    # Load model and images
    m = StyleTranferModel()
    target_photo = load_image(photo_path)
    target_features = m.forward(target_photo)

    target_painting = load_image(painting_path)
    target_grams = {key : compute_gram_matrix(t) for key, t in m.forward(target_painting).items()}

    # Create noisy image which will become style transferred photo
    image = generate_noisy_image(target_photo.shape)
    image.requires_grad = True

    # Create optimizer
    optimizer = torch.optim.Adam([image], lr = 0.1)

    for i in range(epoch_size * iters):
        optimizer.zero_grad() # reset grads
        
        # forward pass
        output = m.forward(image)

        c_loss = content_loss(output, target_features)
        s_loss = style_loss(output, target_grams)
        loss = (alpha * c_loss) + (beta * s_loss)

        # backward pass
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.data.clamp_(0, 1)

        if i % epoch_size == 0:
            print(f"Iter: {i:05}, Content Loss: {c_loss}, Style Loss: {s_loss}. Saving image...")
            checkpoint(image, photo_path, painting_path, alpha, beta, i)

    checkpoint(image, photo_path, painting_path, alpha, beta, i + 1)
    display_image(image)

if __name__ == "__main__":
    transfer_style("photos/neckarfront.jpg", "paintings/starry_night.jpg", iters = 100, alpha = 1, beta = 1000)