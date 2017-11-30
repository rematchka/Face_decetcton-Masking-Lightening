

I=imread('download.jpg');

J = colorConstancy(I, 'modified white patch', 200);
figure
imshow(J);