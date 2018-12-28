orgPic=imread('dog0.jpg');
%figure(1);
logo_width=500;
logo_width_thick=30;
logo_height=600;
logo_height_thick=20;
%logo_colr=[100,100,100];
R0=100;
G0=110;
B0=120;

%start location
width_loc=10;
height_loc=20;
b=orgPic;
%for i=1:mov.numberofframes
%[x,y,z]=size(b);
%height_loc=y-logo_height;
    
    %add advertisement
    %b(width_loc:logo_width+width_loc-1,height_loc:logo_height+height_loc-1,:)=logo_colr; 
size1=ones(logo_width_thick,logo_width);
b(height_loc:logo_width_thick+height_loc-1, width_loc:width_loc+logo_width-1,:)=cat(3,size1*R0,size1*G0,size1*B0); %∫·œﬂ1

%b(width_loc+logo_height-logo_height_thick:width_loc+logo_width_thick-1, height_loc:height_loc+logo_width-1,:)=cat(3,size1*R0,size1*G0,size1*B0); %∫·œﬂ2
imshow(b);
