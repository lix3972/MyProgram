a2=imread('sence1.jpg');
a2_rsz=imresize(a2,[512,680],'lanczos3');
imwrite(a2_rsz,'inputs1.png');
