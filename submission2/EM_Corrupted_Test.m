clc;
clear;

q = 40;
itr = 10;

% read image and add the mask

X = imread('London_256.jpg');
X = rgb2gray(X);
X = im2double(X);

% X = imresize(X,0.4);
img = X;

% img_corrupted = imnoise(img,'salt & pepper',0.02);
% img_corrupted = imnoise(img,'speckle',0.03);
% img_corrupted = imnoise(img,'gaussian',0,0.03);

fprintf(1, '%d corrupted entries\n', nnz(isnan(img_corrupted)));

% create a matrix X from overlapping patches
ws = 8; % window size
patch_factor = 0.5;	% Never set it > 1
no_patches = size(img, 1) / ws;
X = zeros(no_patches^2, (ws/patch_factor)^2);
k = 1;
for i = (1:no_patches*(2*patch_factor)-1)
    for j = (1:no_patches*(2*patch_factor)-1)
        r1 = 1+(i-1)*ws/(2*patch_factor):(i+1)*ws/(2*patch_factor);
        r2 = 1+(j-1)*ws/(2*patch_factor):(j+1)*ws/(2*patch_factor);
        patch = img_corrupted(r1, r2);
        % str1 = strcat('Dump/Img',int2str(k),'.jpg');
        % imwrite(patch,str1);
        X(k,:) = patch(:);
        k = k + 1;
    end
end

% apply Probabilistic PCA with EM
tic
[W,sigma,M,mean,x_t] = EM(X,q,itr);
toc

L = W*inv(W'*W)*x_t;

for i = 1:size(X,2)
   
    L(:,i) = L(:,i) + mean;
    
end

% reconstruct the image from the overlapping patches in matrix L
img_reconstructed = zeros(size(img));
img_noise = zeros(size(img));
k = 1;
for i = (1:no_patches*(2*patch_factor)-1)
    for j = (1:no_patches*(2*patch_factor)-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L(k,:), ws/patch_factor, ws/patch_factor);
        r1 = 1+(i-1)*ws/(2*patch_factor):(i+1)*ws/(2*patch_factor);
        r2 = 1+(j-1)*ws/(2*patch_factor):(j+1)*ws/(2*patch_factor);
        img_reconstructed(r1, r2) =  img_reconstructed(r1, r2) + 0.25*patch;
        k = k + 1;
    end
end
img_final = img_reconstructed;

% show the results
figure;
subplot(2,2,1), imshow(img_corrupted), title('Corrupted image');
str1 = strcat('Dump/Corrupted image','.jpg');
imwrite(img_corrupted,str1);

subplot(2,2,2), imshow(img_final), title('Recovered image');
str1 = strcat('Dump/Recovered image','.jpg');
imwrite(img_final,str1);

subplot(2,2,3), imshow(img_reconstructed), title('Recovered low-rank');

error = img - img_final;
RMS_error = sqrt(sum(sum(error.^2))/(size(X,1)*size(X,2)));

fprintf(1, 'rank(L)=%d\terr=%f\n', rank(L), RMS_error);
