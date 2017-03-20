clc;
clear;

%Image file reading

X = imread('London.jpg');
orig = X;
% X = imnoise(X,'salt & pepper',0.02);
X = rgb2gray(X);
X = im2double(X);

orig = rgb2gray(orig);
orig = im2double(orig);

% X = imresize(X,0.4);
% X(2:30:end) = NaN;


q = 200;
itr = 21;

[W,sigma,M,mean,x_t] = EM(X,q,itr);

rec_image = W*inv(W'*W)*x_t;

for i = 1:size(X,2)
   
    rec_image(:,i) = rec_image(:,i) + mean;
    
end

error = orig - rec_image;

RMS_error = sqrt(sum(sum(error.^2))/(size(X,1)*size(X,2)));
norm_error = norm(error);

str1 = strcat('Recovered_Missing_EM, q=',int2str(q),', itr=',int2str(itr),'.jpg');

imwrite(rec_image,str1);
imwrite(X,'Original_Missing_30.jpg');

disp('RMS Error :');
disp(RMS_error);

disp('Norm Error :');
disp(norm_error);

figure(1);
imshow(X);
title('Original Image');

figure(2);
imshow(rec_image);
title('Reconstructed Image');