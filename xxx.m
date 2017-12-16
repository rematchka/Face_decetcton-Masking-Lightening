

I=imread('24883079_1504597986284107_2044387562_o.jpg');

J = colorConstancy(I, 'progressive', 10,50);
figure
imshow(J);