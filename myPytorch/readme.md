### 1.剪切图片
import torchvision.transforms as transforms  
from PIL import Image  
img = Image.open(A_path).convert('RGB')  # 读取图片img.size=w×h ，img.mode='RGB'  
img.show()  # 显示图片
transforms.CenterCrop((256,256))(img)  # 图片中心剪切  
image.crop(x_start,y_start,x_end,y_end)  # x,y可超出img.size,超出部分的值为黑色0。注：图片左上角为(0,0),即(0,0)处的值不为0。  
###### 示例：PILtry.py和LocationCrop.py  
transforms的二十二个方法：https://blog.csdn.net/u011995719/article/details/85107009  
PIL.Image和np.ndarray图片与Tensor之间的转换：https://blog.csdn.net/tsq292978891/article/details/78767326  
### 2.数据格式
PIL与CV数据格式：https://www.cnblogs.com/ocean1100/p/9494640.html;   
PIL(Python Imaging Library)是Python中最基础的图像处理库。    
PIL图像(w×h)在转换为numpy.ndarray后，格式为(h,w,c)，像素顺序为RGB，其中h=高,w=宽；  
OpenCV在cv2.imread()后数据类型为numpy.ndarray，格式为(h,w,c)，像素顺序为BGR。  

### 1.文件夹说明：
  (1)存放程序  
    data/        :数据相关操作，包括数据预处理、dataset读取等  
    models/      :模型定义。  
    utils/       :可能用到的工具函数  
  (2)存放数据  
    checkpoints/ :保存训练好的模型。  
    datasets/    :存放训练集和测试集。  
    results/     :存放最终测试结果(test.py 输出路径)  
    trainresults/:存放训练时部分测试结果(train.py 训练的一定程度时测试一下结果)  
