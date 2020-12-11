import os
from os import walk
from PIL import Image
import glob
import natsort

fp_in = './results/*.png'
fp_out = 'result.gif'

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def save_gif():
    images = os.listdir('results')
    images = natsort.natsorted(images)
    print(images)




if __name__ == '__main__':
    save_gif()