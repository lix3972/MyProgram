from scipy import ndimage
from numpy import *
from PIL import Image
from pylab import *
im = array(Image.open('adidas.jpg').convert('L'))
# H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
H = array([[1,0,0],[0,1,0],[0,0,1]])
# im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))
im2 = ndimage.affine_transform(im,H)
imshow(im)
figure()
gray()
imshow(im2)
show()