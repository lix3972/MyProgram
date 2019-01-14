### 1.剪切图片
import torchvision.transforms as transforms  
from PIL import Image  
img = Image.open(A_path).convert('RGB')  # 读取图片img.size=w×h ，img.mode='RGB'  
transforms.CenterCrop((256,256))(img)  # 图片中心剪切  
image.crop(x_start,y_start,x_end,y_end)  # x,y可超出img.size,超出部分的值为黑色0。注：图片左上角为(0,0),即(0,0)处的值不为0。  
###### 示例：PILtry.py和LocationCrop.py  
