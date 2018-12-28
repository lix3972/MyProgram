%http://blog.stevenwang.name/image-centroid-217001.html
% 下面为一个实例,供参考
f=magic(10);
g=f>mean(f(:));
g(8,1)=0;
% 通过regionprops获得质心
props=regionprops(g,'Centroid');
% 由help得Centroid的解释为:
% Note that the first element of Centroid is the horizontal coordinate 
% (or x-coordinate) of the center of mass, and the second element is 
% the vertical coordinate (or y-coordinate)
% 得到centroid=[5.5918    4.9388],其中第一个元素为x方向(即列),第二个元素为
% 方向(即行)
centroid=props.Centroid

% 通过一阶中心矩计算质心
m00=sum(g(:));
m10=0;for i=1:10,for j=1:10,m10=m10+i*g(i,j);end,end
m01=0;for i=1:10,for j=1:10,m01=m01+j*g(i,j);end,end
x=m01/m00;
y=m10/m00;
% 得到的[x,y]=[5.5918    4.9388]
[x,y]

% 误差为0
norm([x y]-centroid)