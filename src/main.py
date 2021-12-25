import image_utils
import my_models
from PIL import Image
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

def main():

    #device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    device = "cpu"
    loader = image_utils.simple_loader(device, 255)
    ####################################
    # Images
    ####################################
    original_img = image_utils.load_transform_image('../images/chipys.jpg', loader, device)
    style_img = image_utils.load_transform_image('../images/the-great-wave-off-kanagawa-4-1366×768.jpg', loader, device)
    # Noise
    #generated_img = torch.randn(original_image.shape, device=device, requires_grad=True)
    # Grad is as the net is going to be freeze except for this
    generated_img = original_img.clone().requires_grad_(True)
    ####################################
    # Hyperparameters
    ####################################
    epochs = 2000
    lr = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated_img], lr=lr)
    # Select this features as in the paper
    vgg_feature_layers = [0, 5, 10, 19, 28]

    ####################################
    # Evaluating
    ####################################
    model = my_models.VGG(vgg_feature_layers).to(device).eval()
    for epoch in range(epochs):
        # Obtain features
        generated_f = model(generated_img)
        original_f = model(original_img)
        style_f = model(style_img)
        # Losses
        style_loss = original_loss = 0
        for g_f,         o_f,        s_f in zip(
            generated_f, original_f, style_f):
            # Batch, channel, heigh, width
            B, C, H, W = g_f.shape
            original_loss += torch.mean((g_f-o_f)**2)
            
            # Gram Matrix
            # Multiplies every pixel from every channel with every other pixel
            # For the style
            G = g_f.view(C, H*W) @ g_f.view(C, H*W).T

            A = s_f.view(C, H*W) @ s_f.view(C, H*W).T

            style_loss += torch.mean((G-A)**2)

        total_loss = alpha*original_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ####################################
        # Saving images
        ####################################
        if epoch % 10 == 0:
            print(f"{epoch}/{epochs}")
        if epoch % 200 == 0:
            print(f"Epoch {epoch}| Loss: {total_loss}") 
            torchvision.utils.save_image(generated_img, f"../output/output_{epoch}.png")

if __name__ == "__main__":
    main()