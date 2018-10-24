import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
#read picture
image_raw_data=tf.gfile.FastGFile("/home/t907/ST_GAN/spatial-transformer-GAN-master/glasses/dataset/timg.jpeg",'rb').read()

with tf.Session() as sess :
	img_data=tf.image.decode_jpeg(image_raw_data)
	plt.imshow(img_data.eval())
	resized=tf.image.resize_images(img_data,[144,144],method=0)
	resized_uint8=np.asarray(resized.eval(),dtype='uint8')
	encode_image=tf.image.encode_png(resized_uint8)
	with tf.gfile.GFile("pic144","wb") as f:
		f.write(encode_image.eval())