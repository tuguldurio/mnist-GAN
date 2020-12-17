import base64
import torch
import random
from torchvision.utils import save_image
import model
import utils

from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    # load data
    train_img = next(iter(utils.load_data(1, 64)))[0]
    train_img_path = 'main/static/train_image.jpg'
    save_image(train_img, train_img_path)

    # load model
    G = model.Generator(100, 64)
    G.load_state_dict(torch.load('model/Generator.pth'))
    G.eval()
    G_path = 'main/static/G_image.jpg'
    save_image(G(torch.randn(1, 100, 1, 1)), G_path)

    with open(train_img_path, 'rb') as image_file:
        train_img = base64.b64encode(image_file.read()).decode('utf-8')

    with open(G_path, 'rb') as image_file:
        G_img = base64.b64encode(image_file.read()).decode('utf-8')
    
    # answer for generated image
    answer = 1
    # initialize array of images
    images = [G_img, train_img]
    if random.randrange(2):
        images = reversed(images)
        answer = 2

    return render(request, 'index.html', {'images': images, 'answer': answer})