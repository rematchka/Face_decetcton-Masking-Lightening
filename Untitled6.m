img1=imread('download.jpg');

img2=imread('download (1).jpg');

faceDetector = vision.CascadeObjectDetector();
bbox            = step(faceDetector, img1);

% Draw the returned bounding box around the detected face.
img1 = insertShape(img1, 'Rectangle', bbox);
figure; imshow(img1); title('Detected face');

bbox2            = step(faceDetector, img2);

% Draw the returned bounding box around the detected face.
img2 = insertShape(img2, 'Rectangle', bbox2);
figure; imshow(img2); title('Detected face');


disp(bbox);

x=bbox(1:1);
disp(x);
y =bbox(2:2);
disp(y);
w=bbox(3:3);
disp(w);
h=bbox(4:4);
disp(h);
x2=x+w;
y2=y+h;
mat2 = zeros(size(img1));

disp(size(img1));
disp(size(mat2));
mat2(x:x2, y:y2,:) = img1(x:x2, y:y2,:);
figure
imshow(mat2);
mat2=rgb2gray(mat2);
img1=rgb2gray(img1);
mat3 = zeros(size(img1));
mat3=img2.*img1;