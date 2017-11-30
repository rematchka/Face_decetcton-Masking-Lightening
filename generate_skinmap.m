
    filename='IMG_20161103_154822.jpg';
   
    
    %Read the image, and capture the dimensions
    img_orig = imread('received_1633549073350455.jpeg');
    
    
    
    height = size(img_orig,1);
    width = size(img_orig,2);
    
    %Initialize the output images
    out = img_orig;
    bin = zeros(height,width);
    
    %Apply Grayworld Algorithm for illumination compensation
    
   
%Color Balancing using the Gray World Assumption
%   I - 24 bit RGB Image
%   out - Color Balanced 24-bit RGB Image
%
%   Gaurav Jain, 2010.

    out = uint8(zeros(size(img_orig,1), size(img_orig,2), size(img_orig,3)));
    
    %R,G,B components of the input image
    R = img_orig(:,:,1);
    G = img_orig(:,:,2);
    B = img_orig(:,:,3);

    %Inverse of the Avg values of the R,G,B
    mR = 1/(mean(mean(R)));
    mG = 1/(mean(mean(G)));
    mB = 1/(mean(mean(B)));
    
    %Smallest Avg Value (MAX because we are dealing with the inverses)
    maxRGB = max(max(mR, mG), mB);
    
    %Calculate the scaling factors
    mR = mR/maxRGB;
    mG = mG/maxRGB;
    mB = mB/maxRGB;
   
    %Scale the values
     out(:,:,1) = R*mR;
     out(:,:,2) = G*mG;
     out(:,:,3) = B*mB;
figure; imshow(out);
    img = out;    
    
    %Convert the image from RGB to YCbCr
    img_ycbcr = rgb2ycbcr(img);
    Cb = img_ycbcr(:,:,2);
    Cr = img_ycbcr(:,:,3);
    
    %Detect Skin
    [r,c,v] = find(Cb>=77 & Cb<=127 & Cr>=133 & Cr<=173);
    numind = size(r,1);
    
    %Mark Skin Pixels
    for i=1:numind
        out(r(i),c(i),:) = [0 0 255];
        bin(r(i),c(i)) = 1;
    end
    imshow(img_orig);
    figure; imshow(out);
    figure; imshow(bin);

