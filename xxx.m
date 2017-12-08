

I=imread('24883079_1504597986284107_2044387562_o.jpg');

J = colorConstancy(I, 'single scale retinex', 1.78,40, 20);
figure
imshow(J);