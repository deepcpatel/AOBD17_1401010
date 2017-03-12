% clc
% clear all
% close all

%Image file reading 
X = imread('London.jpg');
X = rgb2gray(X);
X = im2double(X);
X = imresize(X,0.4);

q = 100;
itr = 4;

[W,sigma,M,mean,x_t] = EM(X,q,itr);

rec_image = W*inv(W'*W)*M*x_t;

for i = 1:size(X,2)
   
    rec_image(:,i) = rec_image(:,i) + mean;
    
end

error = X - rec_image;

RMS_error = sqrt(sum(sum(error.^2))/(size(X,1)*size(X,2)));

disp('RMS Error :');
disp(RMS_error);

figure(1);
imshow(X);
title('Original Image');

figure(2);
imshow(rec_image);
title('Reconstructed Image');

% Iteration   RMS_error
%    0         0.1536
%    1         0.0101
%    2         0.0089   % Now onwards the Error remains constant
%    :         0.0089
%    :         0.0089