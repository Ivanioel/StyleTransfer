import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def simple_loader(device, img_size):
    loader = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[]),
    ])
    return loader


def load_transform_image(image_name, loader, device=None, path='./'):
    image_path = path + image_name
    image = Image.open(image_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    image = loader(image).unsqueeze(0).to(device)
    return image
