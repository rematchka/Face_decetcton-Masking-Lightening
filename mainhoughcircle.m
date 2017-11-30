 img = imread('download (3).jpg');
      imshow(img);
      img=rgb2gray(img);
       imgBW = edge(img);
       rad = 24;
       [y0detect,x0detect,Accumulator] = houghcircle1(imgBW,rad,rad*pi);
       figure;
       imshow(imgBW);
       title('edge detection')
       hold on;
       plot(x0detect(:),y0detect(:),'x','LineWidth',2,'Color','yellow');
       figure;
       imagesc(Accumulator);