%% ��ȡ��Ƶ�ļ� �ֽ�Ϊ֡
% [filename,pathname]=uigetfile('*.*','choose a video');
% path = [pathname filename];
% xyloObj = VideoReader(path);
% start = 1;
% nFrames = xyloObj.NumberOfFrames; %��ȡ��Ƶ��֡��
% for k = start :nFrames %����ÿһ֡ 
%     b1 = read(xyloObj, k);
%     b1 = imresize(b1,[240,425]);% �任ͼ������ش�С
%     imwrite(b1,strcat('03052\',num2str(k),'.jpg'),'jpg');%'03052\'Ϊ��ǰĿ¼�µ�03052�ļ���
% end

%%ͼ��ϳ���Ƶ
path = 'C:\lix_Dr\ѧϰ�о�\���볡��\����Ƶ\Ч��\5\';                  %'whiteCar2\'Ϊ��ǰĿ¼�µ�whiteCar2�ļ���
writerObj = VideoWriter('result5mat.avi');   %�����ɵ���Ƶ����Ϊ����Ϊ'car.avi'����Ƶ
open(writerObj);
for i = 500001:500300   
    frame = imread(strcat(path,num2str(i),'.jpg'));%���ļ����ж�ȡͼ��  
    writeVideo(writerObj,frame);
end
close(writerObj);