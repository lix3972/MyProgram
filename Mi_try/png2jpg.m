%�����޸��ļ���
%path1:�޸��ļ��� ��Ӧ·��
%extName:�ļ���չ��
clear;
clc;
close all;
path0=pwd;  %��ȡ��ǰmatlab·������Ϊ�����л�ı䵱ǰ·��������α���Ϊ�˳������ԭ����ǰ·����
path1='C:\lix_Dr\ѧϰ�о�\���볡��\����Ƶ\Ч��\5_painting';%'C:\try\tryb'; %��Ҫ�޸��ļ�����·����
path2='C:\lix_Dr\ѧϰ�о�\���볡��\����Ƶ\Ч��\5_p_jpg\'; %�޸��ļ�������
extName='.jpg';     %�ļ���չ��


eval(['cd ',path1]); %��Ϊrename�����в�����·������Ҫ�л���ǰĿ¼��ָ��·��
n=length(extName);  %Ϊ�˸���չ����ͳ����չ��(��.)����
fileN=dir(path1);   %��ȡ�ļ���Ϣ
fileN=fileN(3:end); 
len=length(fileN);
%%ȥ���ļ���
ni0=0;
for i=1:len
    if ~fileN(i).isdir
        ni0=ni0+1;
        files(ni0)=fileN(i);
    end
end

%% ���ļ���
% ni=0;
for n0=1:length(files)
    oldName=files(n0).name;
    a=imread(oldName);
    a=imresize(a,[360,540],'lanczos3');
    newName=[path2,oldName(1:end-4),'.jpg'];
    imwrite(a,newName);
    
end
eval(['cd ',path0]);
