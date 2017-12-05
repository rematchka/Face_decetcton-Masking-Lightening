if true
i=imread('hellllooo.jpg');
subplot(4,4,1)
imshow(i)
title('original image');
iycbcr=rgb2ycbcr(i);
iycbcr = im2double(iycbcr);
subplot(4,4,2)
imshow(iycbcr)
title('YCBCR space');
y=iycbcr(:,:,1);
cb=iycbcr(:,:,2);
cr=iycbcr(:,:,3);
ccb=cb.^2;
subplot(4,4,3)
imshow(ccb)
title('CB^2');
ccr=(1-cr).^2;
subplot(4,4,4)
imshow(ccr)
title('(1-CR)^2');
cbcr=ccb./cr;
subplot(4,4,5)
imshow(cbcr)
title('CB/CR');
igray=rgb2gray(i);
subplot(4,4,6)
imshow(igray)
title('Gray space');
g=1./3;
l=g*ccb;
m=g*ccr;
n=g*cbcr;

eyemapchr=l+m+n;
subplot(4,4,7)
imshow(eyemapchr)
title('Chrom Eyemap');
J=histeq(eyemapchr);
subplot(4,4,8)
imshow(J)
title('Equalized image');
SE=strel('disk',9,8);
o=imdilate(igray,SE);
p=1+imerode(igray,SE);
eyemaplum=o./p;
subplot(4,4,9)
imshow(eyemaplum)
title('Lum Eyemap');
 eyemapf=double(eyemapchr ).*double( eyemaplum);
 figure
 imshow(eyemapf);
end