import numpy as np
from PIL import Image
import PIL.Image


#maps length of data-item to its originl size
dimensions = {784 : (28, 28), 560 : (20, 28)}
#whether the numpy arrays should be converted to column major or row major form
major = {784 : 'C', 560 : 'F'}
#how much the image should be rotated by:
rotation = {784 : 0, 560 : -90}

#converts the data contained in x to an image. Assumes it is from MNIST or freyface dataset
def save_image(x, filename) :
    assert x.size in dimensions.keys()
    assert filename.endswith('jpg')

    dimension = dimensions[x.size]
    x = np.copy(x).reshape(dimension, order=major[x.size])

    image = Image.fromarray((1 - x) * 255).convert('RGB').rotate(rotation[x.size])
    image.save(filename)

    return image

def multipleImages(image):
    horizontal = []
    #for ii in [0.1, 0.15, 0.2, 0.25, 0.3, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for ii in range(10):
        list_im = []
        #for jj in [0.1, 0.15, 0.2, 0.25, 0.3, 0.75, 0.8, 0.85, 0.9, 0.95]:
        for jj in range(10):
            # list_im.append(image + str(ii) + str(jj) + '.jpg')
            list_im.append(image + str(ii) + str(jj) + '.jpg')
        imgs    = [ PIL.Image.open(i) for i in list_im ]
        imgs_comb = np.hstack( (np.asarray( i ) for i in imgs ) )
        horizontal.append(imgs_comb)

    imgs_comb = np.vstack( (np.asarray( i ) for i in horizontal ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(image + '.jpg')
