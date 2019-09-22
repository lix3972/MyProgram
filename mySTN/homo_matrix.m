clear
syms rz ry rx sx sy tx ty
h_rx=[cos(rz), -sin(rz), 0;sin(rz), cos(rz), 0; 0, 0, 1];
h_ry=[cos(ry), 0, -sin(ry); 0,1,0; sin(ry),0,cos(ry)];
h_rz=[1,0,0; 0,cos(rx), -sin(rx); 0,sin(rx),cos(rx)];
h_s=[sx,0,0; 0,sy,0; 0,0,1];
h1 = h_s*h_rx*h_ry*h_rz;
h2 = h_s*h_ry*h_rz*h_rx;
h1
h2