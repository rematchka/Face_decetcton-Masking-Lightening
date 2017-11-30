I=imread('received_1633549073350455.jpeg');
hsvImage = rgb2hsv(I);
sImage = hsvImage(:,:,2);
mask = sImage > 0.1; % Or whatever value works.
% Mask the image using bsxfun() function
maskedRgbImage = bsxfun(@times, I, cast(mask, 'like', I));
imshow(I);