#为了调整戴眼镜的初始位置
import matplotlib.pyplot as plt
glassesX=[]
glassesY=[]
glassesXY=[]
y=0
for x in range(11):
    glassesXY.append([x,y])
for x in range(20,-1,-1):
    y=int((100-(x-10)**2)**0.5)
    glassesXY.append([x-10, y])
for x in range(21):
    y=-int((100-(x-10)**2)**0.5)
    glassesXY.append([x-10, y])
for x,y in glassesXY:
    glassesX.append(x)
    glassesY.append(y)
plt.plot(glassesX,glassesY)
plt.show()
