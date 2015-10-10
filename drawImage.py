import numpy as np
from PIL import Image

#draws the files. I found this somewhere on the site thing, so might as well.
def draw_img(row,c):
    pix = row.reshape((28, 28)).astype('uint8')
    im = Image.fromarray(pix)
    im.save('out%d.png' % c)
