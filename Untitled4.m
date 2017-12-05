file1='FB_IMG_1463846848502.jpg';
file2='download.jpg';
file3='FB_IMG_1455471797813.jpg';
file4='20161223_163424.jpg';
file5='mariam.jpg';
I=imread(file5);

ggg=I;
I=double(I);


[hue,s,v]=rgb2hsv(I);

figure()
imshow(ggg);


%fid = fopen('Mymatrix.txt','wt');
%for ii = 1:size(hue,1)
 %   fprintf(fid,'%g\t',hue(ii,:));
  %  fprintf(fid,'\n');
%end
%fclose(fid);
%fid = fopen('Mymatrix1.txt','wt');
%for ii = 1:size(s,1)
 %   fprintf(fid,'%g\t',s(ii,:));
  %  fprintf(fid,'\n');
%end
%fclose(fid);
%fid = fopen('Mymatrix3.txt','wt');
%for ii = 1:size(v,1)
 %   fprintf(fid,'%g\t',v(ii,:));
  %  fprintf(fid,'\n');
%end
%fclose(fid);







cb =  0.148* I(:,:,1) - 0.291* I(:,:,2) + 0.439 * I(:,:,3) + 128;
cr =  0.439 * I(:,:,1) - 0.368 * I(:,:,2) -0.071 * I(:,:,3) + 128;
%fid = fopen('Mymatrix4.txt','wt');
%for ii = 1:size(cb,1)
 %   fprintf(fid,'%g\t',cb(ii,:));
  %  fprintf(fid,'\n');
%end
%fclose(fid);
%fid = fopen('Mymatrix5.txt','wt');
%for ii = 1:size(cr,1)
 %   fprintf(fid,'%g\t',cr(ii,:));
  %  fprintf(fid,'\n');
%end
%fclose(fid);
%disp(cb);
[w, h]=size(I(:,:,1));
segment=zeros(w,h);
im=zeros(w,h,3);
for i=1:w
    for j=1:h            
        if  140<=cr(i,j) && cr(i,j)<=165 && 140<=cb(i,j) && cb(i,j)<=195 && 0.01<=hue(i,j) && hue(i,j)<=0.1     
            segment(i,j)=1; 
            
        else       
            segment(i,j)=0;    
        end    
    end
end
i1=I(:,:,1);


im(:,:,1)=i1.*segment;   
im(:,:,2)=I(:,:,2).*segment; 
im(:,:,3)=I(:,:,3).*segment; 
figure,title('segment');imshow((segment));
%se=strel('disk',11);
%se=strel('ball',5,5);

se=strel('disk',9);
dilate=imdilate(segment,se);
erode=imerode(dilate,se);

[l ne]=bwlabel(erode);
 prpid=regionprops(l);
st = regionprops(l, 'BoundingBox' ,'Area');
for k = 1 : length(st)
  thisBB = st(k).BoundingBox;
  rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...
  'EdgeColor','r','LineWidth',2 )
end
figure,title('dilate');imshow((dilate));

figure,title('erode');imshow((erode));



% Let's extract the second biggest blob - that will be the hand.
allAreas = [st.Area];
[sortedAreas, sortingIndexes] = sort(allAreas, 'descend');
handIndex = sortingIndexes(1); % The hand is the second biggest, face is biggest.
% Use ismember() to extact the hand from the labeled image.
handImage = ismember(l, handIndex) ;
% Now binarize
handImage = handImage > 0;
% Display the image.
figure
imshow(handImage, []);
im1=zeros(w,h,3);
im1(:,:,1)=ggg(:,:,1).*uint8(dilate);  

im1(:,:,2)=ggg(:,:,2).*uint8(dilate); 
im1(:,:,3)=ggg(:,:,3).*uint8(dilate); 

figure
imshow(im1);
maskedRgbImage = bsxfun(@times, ggg, cast(erode, 'like', ggg));
figure
imshow(maskedRgbImage);

YIQ = rgb2ntsc(im1);
YCBCR = rgb2ycbcr(im1);
Ichannel=YIQ(:,:,1);
CR=YCBCR(:,:,3);
CB=YCBCR(:,:,2);
imhhhh=CB+CR;
se=strel('rectangle',[9 9]);
dilate=imdilate(Ichannel,se);
erode=imerode(dilate,se);
figure
imshow(erode);



Eyemap=(1-(double(CR).^2).*(1-double(Ichannel)));
figure
imshow(Ichannel);
immm=Ichannel>=1 ;
disp(Ichannel);
figure
imshow(immm);


imgcanny=edge(cb,'canny');
figure
imshow(imgcanny);










