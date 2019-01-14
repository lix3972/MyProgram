### 1.剪切图片
import torchvision.transforms as transforms  
from PIL import Image  
img = Image.open(A_path).convert('RGB')  # 读取图片img.size=w×h ，img.mode='RGB'  
transforms.CenterCrop((256,256))(img)  # 图片中心剪切  
image.crop(x_start,y_start,x_end,y_end)  # x,y可超出img.size,超出部分的值为黑色0。注：图片左上角为(0,0),即(0,0)处的值不为0。  
###### 示例：PILtry.py和LocationCrop.py  
transforms的二十二个方法：https://blog.csdn.net/u011995719/article/details/85107009  
PIL.Image和np.ndarray图片与Tensor之间的转换：https://blog.csdn.net/tsq292978891/article/details/78767326  
PIL与CV数据格式：https://www.cnblogs.com/ocean1100/p/9494640.html; PIL(Python Imaging Library)是Python中最基础的图像处理库。PIL图像在转换为numpy.ndarray后，格式为(h,w,c)，像素顺序为RGB，其中h=高,w=宽；OpenCV在cv2.imread()后数据类型为numpy.ndarray，格式为(h,w,c)，像素顺序为BGR。
