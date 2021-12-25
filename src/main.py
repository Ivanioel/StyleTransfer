import argparse

import torch
import torch.optim as optim
import torchvision

import image_utils
import my_models


def main():
    ####################################
    # Arguments
    ####################################
    parser = argparse.ArgumentParser(description='Style Transfer from one image to another')
    parser.add_argument('-o', '--original_img', type=str,
                        help='Path from the original image', required=True)
    parser.add_argument('-s', '--style_img', type=str,
                        help='Path from the style image', required=True)
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to try')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Weigth of the original image')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Weigth of the style image')

    parser.add_argument('--optimizer', type=str,
                        default='Adam', choices=['Adam'],
                        help='Optimizer')

    parser.add_argument('--model', type=str,
                        default='VGG19', choices=['VGG19'],
                        help='Backbone model')

    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    device = "cpu"
    loader = image_utils.simple_loader(device, 255)
    ####################################
    # Images
    ####################################
    original_img = image_utils.load_transform_image(
                                args.original_img, loader, device)
    style_img = image_utils.load_transform_image(
                                args.style_img, loader, device)

    # Uncomment this if want to start from noise
    # generated_img = torch.randn(original_image.shape, device=device, requires_grad=True)

    # Grad is necesry as the net is going to be freeze except for this
    generated_img = original_img.clone().requires_grad_(True)
    ####################################
    # Hyperparameters
    ####################################
    epochs = args.epochs
    lr = args.lr
    alpha = args.alpha
    beta = args.beta

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([generated_img], lr=lr)
    else:
        print('Optimizer not in the possible choices')
        exit(0)

    if args.model == 'VGG19':
        # Select this features as in the paper
        vgg_feature_layers = [0, 5, 10, 19, 28]
        model = my_models.VGG(vgg_feature_layers).to(device)
    else:
        print('Model not in the possible choices')
        exit(0)

    ####################################
    # Evaluating
    ####################################
    model.eval()
    for epoch in range(epochs+1):
        # Obtain features
        generated_f = model(generated_img)
        original_f = model(original_img)
        style_f = model(style_img)
        # Losses
        style_loss = original_loss = 0
        for g_f, o_f, s_f in zip(
            generated_f, original_f, style_f
        ):
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
        # I get desesperated if I have to execute over CPU
        if device == 'cpu' and epoch % 10 == 0:
            print(f"{epoch}/{epochs}")
        if epoch % 200 == 0:
            print(f"Epoch {epoch}| Loss: {total_loss}")
            torchvision.utils.save_image(generated_img, f"../output/output_{epoch}.png")


if __name__ == "__main__":
    main()
