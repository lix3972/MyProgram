%http://blog.stevenwang.name/image-centroid-217001.html
% ����Ϊһ��ʵ��,���ο�
f=magic(10);
g=f>mean(f(:));
g(8,1)=0;
% ͨ��regionprops�������
props=regionprops(g,'Centroid');
% ��help��Centroid�Ľ���Ϊ:
% Note that the first element of Centroid is the horizontal coordinate 
% (or x-coordinate) of the center of mass, and the second element is 
% the vertical coordinate (or y-coordinate)
% �õ�centroid=[5.5918    4.9388],���е�һ��Ԫ��Ϊx����(����),�ڶ���Ԫ��Ϊ
% ����(����)
centroid=props.Centroid

% ͨ��һ�����ľؼ�������
m00=sum(g(:));
m10=0;for i=1:10,for j=1:10,m10=m10+i*g(i,j);end,end
m01=0;for i=1:10,for j=1:10,m01=m01+j*g(i,j);end,end
x=m01/m00;
y=m10/m00;
% �õ���[x,y]=[5.5918    4.9388]
[x,y]

% ���Ϊ0
norm([x y]-centroid)