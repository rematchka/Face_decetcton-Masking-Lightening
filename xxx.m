

I=imread('24883079_1504597986284107_2044387562_o.jpg');

J = colorConstancy(I, 'modified white patch', 178,40, 20);
figure
imshow(J);