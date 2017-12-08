I=imread('ssss.png');
hsv=rgb2hsv(I);
I=double(I);
%I=imread(frame);

img = rgb2ycbcr(I);
y = double(img(:,:,1));
Cb = double(img(:,:,2));
fid = fopen('cbb.txt','wt');
for ii = 1:size(Cb,1)
   fprintf(fid,'%g\t',Cb(ii,:));
  fprintf(fid,'\n');
end

Cr = double(img(:,:,3));
fid = fopen('crr.txt','wt');
for ii = 1:size(Cr,1)
   fprintf(fid,'%g\t',Cr(ii,:));
  fprintf(fid,'\n');
end


x=Cr.^2;
fid = fopen('x.txt','wt');
for ii = 1:size(x,1)
   fprintf(fid,'%g\t',x(ii,:));
  fprintf(fid,'\n');
end

l=sum(sum(x));
disp(l);
ll=Cr/Cb;
fid = fopen('div.txt','wt');
for ii = 1:size(ll,1)
   fprintf(fid,'%g\t',ll(ii,:));
  fprintf(fid,'\n');
end

w=sum(sum(Cr/Cb));

 eitha=0.95*(l/w);
 disp(eitha);
 Mouthmap=(Cr.^2).*(((Cr.^2)-eitha*(Cr./Cb)).^2);
 EnLip=hsv(:,:,2).*Mouthmap;
Mouthmap=Mouthmap/ max (Mouthmap(:));

figure
EnLip=EnLip/ max (EnLip(:));
imshow(EnLip);title('EN lip enhancment');
se = strel('disk',5);
disp(se);



tophatFiltered = imtophat(EnLip,se);
figure
imshow(tophatFiltered);title('EN lip Top hat')
level = graythresh(tophatFiltered);
BW = tophatFiltered > level;

figure
imshow(BW);title('thersholded img')
figure
 imshow(Cr.^2);
 figure
 imshow(Mouthmap);
 

 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 img = rgb2ycbcr(I);
 y = double(img(:,:,1));
Cb = double(img(:,:,2));
Cr = double(img(:,:,3));

Cr = uint8(255*mat2gray(Cr));

Cb = uint8(255*mat2gray(Cb));
 eitha=0.95*((sum(sum(Cr.^2)))/sum(sum(Cr./Cb)));
 Mouthmap=(Cr.^2).*(((Cr.^2)-eitha*(Cr./Cb)).^2);
%Mouthmap=Mouthmap/ max (Mouthmap(:));



 figure
 imshow(Mouthmap);
 
 
 
 


