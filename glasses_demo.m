%����ͼƬ����˳�����glassesXY������
pause(2);
i=1;
for x = 0:10
    glassesXY(i,1)=x;
    glassesXY(i,2)=0;
    i=i+1;
end
for x =20:-1:0 %in range(20,-1,-1):
    glassesXY(i,1)=x-10;
    glassesXY(i,2)=floor((100-(x-10)^2)^0.5);
    i=i+1;
end    
for x =0:20 %in range(21):
    glassesXY(i,1)=x-10;
    glassesXY(i,2)=-floor((100-(x-10)^2)^0.5);
    i=i+1;
end
%��ͼƬ
pathInp='D:\glassesDemo\input\image_g0_input';%+xy.png
pathOut='D:\glassesDemo\output\image_g0_output';
for j=1:53
    x=glassesXY(j,1);
    y=glassesXY(j,2);
    pathImgInp=[pathInp,num2str(x),num2str(y),'.png'];
    pathImgOut=[pathOut,num2str(x),num2str(y),'.png'];
    imgInp=imread(pathImgInp);
    imgOut=imread(pathImgOut);
    subplot(121),imshow(imgInp);
    title('��ʼλ��','Color','y');
    subplot(122),imshow(imgOut);
    title('���������','Color','y');
    pause(0.15);
end
    
    

