% a=[1,2; 3,4];
% b=[5;6];
% x =  a\b;
% a_inv = inv(a);
syms a b c d x1 x2 y1 y2 y x A 
% y = [a,b;c,d]*[x1;x2];
A = [a,b;c,d];
x = inv(A) * [y1; y2];
x1 = x(1);
x2 = x(2);
%det_a = diff(x1, a);
fjac_a=simplify(jacobian(x, a));
fjac_b=simplify(jacobian(x, b));
fjac_c=simplify(jacobian(x, c));
fjac_d=simplify(jacobian(x, d));

a=1;b=2;c=3;d=4;y1=5;y2=6;
eval(fjac_a)
