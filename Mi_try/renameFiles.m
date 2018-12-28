%批量修改文件名
%path1:修改文件名 对应路径
%extName:文件扩展名
clear;
clc;
close all;
path0=pwd;  %获取当前matlab路径。因为程序中会改变当前路径，定义次变量为了程序最后还原到当前路径。
path1='C:\try\QTDownloadRadio';%'C:\try\tryb'; %需要修改文件名的路径。
pathExcel='c:\try\高手.xlsx'; %修改文件名数据
extName='.mp3';     %文件扩展名


eval(['cd ',path1]); %因为rename参数中不能有路径，需要切换当前目录到指定路径
n=length(extName);  %为了改扩展名，统计扩展名(含.)长度
fileN=dir(path1);   %获取文件信息
fileN=fileN(3:end); 
len=length(fileN);
%%去掉文件夹
ni0=0;
for i=1:len
    if ~fileN(i).isdir
        ni0=ni0+1;
        files(ni0)=fileN(i);
    end
end
%% 导入excel 文件名数据
A=xlsread(pathExcel); %导入数据为数值型，两列。第一列文件名，第二列集数

%% 改文件名
% ni=0;
for n0=1:length(files)
%     if ~files(n0).isdir  %不是文件夹，则修改文件名
%         ni=ni+1;
        oldName=files(n0).name;
        %newName=strcat(oldName(1),,num2str(n0),extName);%strcat(oldName(1:end-n),'高手',num2str(ni),extName);
        ind=find(A==str2double(oldName));
        newName=[num2str(ind),extName];
        cmd=['!rename' 32 strcat(oldName) 32 strcat(newName)];
        eval(cmd);
%     end
end
eval(['cd ',path0]);
