import numpy as np
from PIL import Image

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
