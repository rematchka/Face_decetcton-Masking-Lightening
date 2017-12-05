

I=imread('FB_IMG_1463846848502.jpg');

J = colorConstancy(I, 'single scale retinex', 100);
figure
imshow(J);