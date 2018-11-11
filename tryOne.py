import tensorflow as tf
blockSize=2
H=2
W=2
tag="TRAIN_real"
b=[[[[1,2],[3,4]],[[5,6],[7,8]]],[[[1,2],[3,4]],[[5,6],[7,8]]],[[[1,2],[3,4]],[[5,6],[7,8]]],[[[1,2],[3,4]],[[5,6],[7,8]]]]
#a=tf.batch_to_space(b[:blockSize**2],crops=[[0,0],[0,0]],block_size=blockSize)
imageOne = tf.batch_to_space(b[:blockSize**2],crops=[[0,0],[0,0]],block_size=blockSize)
imagePermute = tf.reshape(imageOne,[H,blockSize,W,blockSize,-1])
imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
imageBlocks = tf.reshape(imageTransp,[1,H*blockSize,W*blockSize,-1])
#summary = tf.summary.image(tag,imageBlocks)
with tf.Session() as sess:
    imageOne_run=sess.run(imageOne)
    imagePermute_run =sess.run(imagePermute)
    imageTransp_run =sess.run(imageTransp)
    imageBlocks_run =sess.run(imageBlocks)
    print('b=',b)
    print('imageOne=',sess.run(imageOne))
    print('imagePermute =',sess.run(imagePermute))
    print('imageTransp=',sess.run(imageTransp))
    print('imageBlocks=', sess.run(imageBlocks))
    print('end')
    # #print('summary=', sess.run(summary))
    #print('', sess.run())
