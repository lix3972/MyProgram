avs0=[3.8,2.95,6,0,12.7,10.6,2.7,4.5];
HEVC0=[13.2,2.5,5.7,11.2,11.8,0,6,11.1];
avs1=[4 4 4 3 0 3 1 1];
HEVC1=[5 4 4 2 0 3 1 1];
avs2=avs1+avs0./15;
hevc=HEVC1+HEVC0./15;
rslt=[avs2' hevc'];
bar(rslt,1);
axis([0.5 8.5 0 10]);