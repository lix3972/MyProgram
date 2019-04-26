im=imread('inputs3.png');
img=double(im);
[x,y]=gradient(img);
hold on
x1=x(:,:,1);
y1=y(:,:,1);
quiver(x1(end:-1:1,:),y1(end:-1:1,:));
hold off;