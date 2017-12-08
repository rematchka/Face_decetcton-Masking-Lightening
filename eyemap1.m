I=imread('ssss.png');
%I=imread(frame);
EyeMap = rgb2ycbcr(I);
temp1=0;
temp2=0;
%Q = rgb2gray(I);
%imshow(EyeMap)
[m n l]=size(EyeMap);
%Finding EyeMapC
y = double(EyeMap(:,:,1));
Cb = double(EyeMap(:,:,2));
Cr = double(EyeMap(:,:,3));
Z=Cr;
fid = fopen('y.txt','wt');
for ii = 1:size(y,1)
   fprintf(fid,'%g\t',y(ii,:));
  fprintf(fid,'\n');
end


Q = Cb.^2;
R = (255-Cr).^2;
G = Cb./Cr;
CrCb = Cr./Cb;

%Eye Map for Crominance
EyeC=(Q+R+G)/3;


CRS = Cr.^2;

ssCRS = sum(sum(CRS));
ssCrCb=sum(sum(CrCb));
eta = 0.95 * ssCRS/ssCrCb;
x= CRS - eta * CrCb;
MM = CRS.*x.*x;

SE=strel('disk',4) ;
disp(SE);
UP=imdilate(y,SE);
Down=imerode(y,SE);
EyeY= UP./(Down);
EyeMap=EyeY.*EyeC;
colormap(gray);

subplot(1,4,1), imagesc(I);title ('Face');axis off;
subplot(1,4,2), imagesc(Q);title ('Cb^2');axis off;
subplot(1,4,3), imagesc(R);title ('(Cr-complement)^2');axis off;
subplot(1,4,4), imagesc(EyeC);title ('Eye-Map-C'); axis off;
colormap(gray);
figure;
subplot(1,4,1), imagesc(UP);title ('Dilation'); axis off;
subplot(1,4,2), imagesc(Down);title ('Erotion'); axis off;
subplot(1,4,3), imagesc(EyeY);title ('EyeY'); axis off;
subplot(1,4,4), imagesc(EyeMap);title ('EyeMap'); axis off;
colormap(gray);

imwrite (EyeC / max (EyeC(:)), 'out.jpg');

EyeC=EyeC/ max (EyeC(:));
figure
imshow(EyeC);

UP=UP/ max (UP(:));
figure
imshow(UP);
EyeMap=EyeMap/ max (EyeMap(:));
figure
imshow(EyeMap);
normalizedImage = uint8(255*mat2gray(EyeMap));
figure
imshow(normalizedImage);title('norm img')
%iiiii=isodata(normalizedImage);
%BW = im2bw(normalizedImage,iiiii);

%figure
%imshow(BW);title('isodata thersholding img')

%normImage = mat2gray(EyeMap);
normImage = im2double(EyeMap);

%iiiii=isodata2(normImage);
%BW = im2bw(normalizedImage,iiiii);

%figure
%imshow(BW);title('isodata thersholding img2')

level = graythresh(normImage);
BW1 = normImage > level;
figure
imshow(BW1);title('isodata thersholding img2')







