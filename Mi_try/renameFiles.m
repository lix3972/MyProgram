%�����޸��ļ���
%path1:�޸��ļ��� ��Ӧ·��
%extName:�ļ���չ��
clear;
clc;
close all;
path0=pwd;  %��ȡ��ǰmatlab·������Ϊ�����л�ı䵱ǰ·��������α���Ϊ�˳������ԭ����ǰ·����
path1='C:\try\QTDownloadRadio';%'C:\try\tryb'; %��Ҫ�޸��ļ�����·����
pathExcel='c:\try\����.xlsx'; %�޸��ļ�������
extName='.mp3';     %�ļ���չ��


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
%% ����excel �ļ�������
A=xlsread(pathExcel); %��������Ϊ��ֵ�ͣ����С���һ���ļ������ڶ��м���

%% ���ļ���
% ni=0;
for n0=1:length(files)
%     if ~files(n0).isdir  %�����ļ��У����޸��ļ���
%         ni=ni+1;
        oldName=files(n0).name;
        %newName=strcat(oldName(1),,num2str(n0),extName);%strcat(oldName(1:end-n),'����',num2str(ni),extName);
        ind=find(A==str2double(oldName));
        newName=[num2str(ind),extName];
        cmd=['!rename' 32 strcat(oldName) 32 strcat(newName)];
        eval(cmd);
%     end
end
eval(['cd ',path0]);
