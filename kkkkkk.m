A = imread('received_1633549073350455.jpeg');
figure
imshow(A,'InitialMagnification',25)
title('Original Image')
A_lin = rgb2lin(A);
percentiles = 10;
illuminant = illumgray(A_lin,percentiles);
B_lin = chromadapt(A_lin,illuminant,'ColorSpace','linear-rgb');
B = lin2rgb(B_lin);
figure
imshow(B,'InitialMagnification',25)
title(['White-Balanced Image Using Gray World with percentiles=[' ...
    num2str(percentiles) ' ' num2str(percentiles) ']']);