file1='FB_IMG_1463846848502.jpg';
file2='download.jpg';
file3='FB_IMG_1455471797813.jpg';
file4='20161223_163424.jpg';

im = imread(file2);
im=rgb2gray(im);
e = edge(im, 'canny');
imshow(e);
radii = 15:1:40;
h = circle_hough(e, radii, 'same', 'normalise');
peaks = circle_houghpeaks(h, radii, 'nhoodxy', 15, 'nhoodr', 21, 'npeaks', 10);
imshow(im);
hold on;
for peak = peaks
    [x, y] = circlepoints(peak(3));
    plot(x+peak(1), y+peak(2), 'g-');
end
hold off