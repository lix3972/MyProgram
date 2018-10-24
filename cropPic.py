import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
#read picture
image_raw_data=tf.gfile.FastGFile("/home/t907/ST_GAN/spatial-transformer-GAN-master/glasses/pic/pic0/000010.png",'rb').read()

with tf.Session() as sess :
	# decode Image to data
	img_data=tf.image.decode_png(image_raw_data)
	#show Image
	plt.imshow(img_data.eval())
	plt.show()
	#resize Image
	#resized=tf.image.resize_images(img_data,[144,144],method=0)
	#resized_uint8=np.asarray(resized.eval(),dtype='uint8')
	
	#crop or pad Image
	croped=tf.image.resize_image_with_crop_or_pad(img_data,144,144)
	#encode data to Image
	encode_image=tf.image.encode_png(croped)
	with tf.gfile.GFile("/home/t907/ST_GAN/spatial-transformer-GAN-master/glasses/dataset/pic144.png","wb") as f:
		f.write(encode_image.eval())