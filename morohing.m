clc
close all

% Providing the Source(im1) and Destination(im2) Images.
im1=imread('images.jpg');
im2=imread('download.jpg');

% Resizing the images if necessary(Both the images should be equal in size) 
im2=imresize(im2,[size(im1,1) size(im1,2)]);
[h w]=size(im2(:,:,1));
im3=zeros(h,w,3);
disp(size(im3));
disp(size(im2));
disp(size(im1));
% Number of intermediate images required
n=100;

% Running the for loop through the rows and columns of the pixel matrix(of the images)
for i = 1:n
    im3(:,:,1)=intermediate(im1(:,:,1),im2(:,:,1),n,i);
    im3(:,:,2)=intermediate(im1(:,:,2),im2(:,:,2),n,i);
    im3(:,:,3)=intermediate(im1(:,:,3),im2(:,:,3),n,i);
    
    imshow(uint8(im3))
    pause(0.1)
end

