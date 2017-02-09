% video RobustPCA example: separates background and foreground
clc;
clear;

movieFile = 'lift_input.mp4';

% open the movie
n_frames = 30;
movie = VideoReader(movieFile);
frate = movie.FrameRate;    
height = movie.Height;
width = movie.Width;
mult = height*width;

% vectorize every frame to form matrix X
X = zeros(mult, n_frames);
i=0;

while hasFrame(movie)
    i = i+1;
    frame = readFrame(movie);
    frame = rgb2gray(frame);
    X(:,i) = reshape(frame,[],1);
    X(:,i) = X(:,i) + double(randi([0,30],mult,1));    % Adding random error to the original video
end

% apply Robust PCA
lambda = 1/sqrt(max(size(X)));
tic
[L,S] = RobustPCA(X, lambda/3, 10*lambda/3, 1e-9);  % Maximum Error Tolerance or delta is 1e-9
toc

% prepare the new movie file
vidObj = VideoWriter('lift_output.avi');
vidObj.FrameRate = frate;
open(vidObj);
range = 255;
map = repmat((0:range)'./range, 1, 3);
S = medfilt2(S, [5,1]); % median filter in time

for i = 1:n_frames
    frame1 = reshape(X(:,i),height,[]);
    frame2 = reshape(L(:,i),height,[]);
    frame3 = reshape(abs(S(:,i)),height,[]);
    % median filter in space; threshold
    frame3 = (medfilt2(abs(frame3), [5,5]) > 5).*frame1;
    % stack X, L and S together
    frame = mat2gray([frame1, frame2, frame3]);
    frame = gray2ind(frame,range);
    frame = im2frame(frame,map);
    writeVideo(vidObj,frame);
end

close(vidObj);
