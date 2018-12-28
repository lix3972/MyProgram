%% 读取视频文件 分解为帧
% [filename,pathname]=uigetfile('*.*','choose a video');
% path = [pathname filename];
% xyloObj = VideoReader(path);
% start = 1;
% nFrames = xyloObj.NumberOfFrames; %获取视频总帧数
% for k = start :nFrames %遍历每一帧 
%     b1 = read(xyloObj, k);
%     b1 = imresize(b1,[240,425]);% 变换图像的像素大小
%     imwrite(b1,strcat('03052\',num2str(k),'.jpg'),'jpg');%'03052\'为当前目录下的03052文件夹
% end

%%图像合成视频
path = 'C:\lix_Dr\学习研究\融入场景\短视频\效果\5\';                  %'whiteCar2\'为当前目录下的whiteCar2文件夹
writerObj = VideoWriter('result5mat.avi');   %将生成的视频保存为名称为'car.avi'的视频
open(writerObj);
for i = 500001:500300   
    frame = imread(strcat(path,num2str(i),'.jpg'));%从文件夹中读取图像  
    writeVideo(writerObj,frame);
end
close(writerObj);