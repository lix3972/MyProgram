%批量修改文件名
%path1:修改文件名 对应路径
%extName:文件扩展名
clear;
clc;
close all;
path0=pwd;  %获取当前matlab路径。因为程序中会改变当前路径，定义次变量为了程序最后还原到当前路径。
path1='C:\lix_Dr\学习研究\融入场景\短视频\效果\5_painting';%'C:\try\tryb'; %需要修改文件名的路径。
path2='C:\lix_Dr\学习研究\融入场景\短视频\效果\5_p_jpg\'; %修改文件名数据
extName='.jpg';     %文件扩展名


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

%% 改文件名
% ni=0;
for n0=1:length(files)
    oldName=files(n0).name;
    a=imread(oldName);
    a=imresize(a,[360,540],'lanczos3');
    newName=[path2,oldName(1:end-4),'.jpg'];
    imwrite(a,newName);
    
end
eval(['cd ',path0]);
