clc;
clear;

%Image file reading 

X = imread('8_1.jpg');
X = rgb2gray(X);
X = im2double(X);

% X = imresize(X,0.4);

q = 100;

[W,sigma,M,mean,x_t] = PPCA(X,q);

rec_image = W*inv(W'*W)*M*x_t;       	% See x_t construction in PPCA function
										% Here M*X_t -> M*inv(M)*W'*x_t, which
										% is redundant.
for i = 1:size(X,2)
   
    rec_image(:,i) = rec_image(:,i) + mean;
    
end

error = X - rec_image;

RMS_error = sqrt(sum(sum(error.^2))/(size(X,1)*size(X,2)));

str1 = strcat('Compressed_PPCA, q=',int2str(q),'.jpg');

imwrite(rec_image,str1);
imwrite(X,'Original.jpg');

disp('RMS Error :');
disp(RMS_error);

figure(1);
imshow(X);
title('Original Image');

figure(2);
imshow(rec_image);
title('Reconstructed Image');