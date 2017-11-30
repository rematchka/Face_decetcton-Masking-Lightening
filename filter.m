img1=imread('download.jpg');
img1=rgb2gray(img1);
img2=imread('download (2).jpg');
[col row]=size(img1);

img2=rgb2gray(img2);
img3=imfilter(img1,img2);
figure
imshow(img3);