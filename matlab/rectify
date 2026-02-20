%% Load original stereo images and predicted masks
I1 = imread('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\left\New folder\Ls4f35_m.JPG');
I2 = imread('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\right\New folder\Rs4f35_m.JPG');

M1 = imread('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\pred_mask_Ls4f35.png');
M2 = imread('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\pred_mask_Rs4f35.png');

%% Load stereo calibration session
S = load('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\calibrationSession.mat');
stereoParams = S.calibrationSession.CameraParameters;

%% Preprocess masks (ensure logical first)
if size(M1,3) > 1, M1 = rgb2gray(M1); end
if size(M2,3) > 1, M2 = rgb2gray(M2); end

M1b = bwareaopen(M1 > 0, 100);
M2b = bwareaopen(M2 > 0, 50);

% Keep the dominant crack component (adjust if you truly need >1)
M1c = bwareafilt(M1b, 1);
M2c = bwareafilt(M2b, 1);

% Morphological cleanup (orientation-agnostic; change if you prefer line SE)
M1c = imclose(M1c, strel('disk', 2));
M2c = imclose(M2c, strel('disk', 2));

M11 = uint8(M1c) * 255;
M22 = uint8(M2c) * 255;

figure;
subplot(1,2,1); imshow(M1);  title('Raw predicted mask (L)');
subplot(1,2,2); imshow(M11); title('Cleaned mask (L)');

figure;
subplot(1,2,1); imshow(M2);  title('Raw predicted mask (R)');
subplot(1,2,2); imshow(M22); title('Cleaned mask (R)');

%% Convert to logical for rectification
M11 = M11 > 0;
M22 = M22 > 0;

%% Rectify RGB and masks (same OutputView everywhere)
outputView = 'valid';

[J1, J2, Q] = rectifyStereoImages(I1, I2, stereoParams, 'OutputView', outputView);

% For masks: nearest-neighbor interpolation
[J1m, J2m] = rectifyStereoImages(M11, M22, stereoParams, 'nearest', 'OutputView', outputView);
J1m = uint8(J1m) * 255;
J2m = uint8(J2m) * 255;

figure;
subplot(1,2,1); imshow(stereoAnaglyph(J1, J2));  title('Rectified RGB anaglyph');
subplot(1,2,2); imshow(stereoAnaglyph(J1m, J2m)); title('Rectified masks anaglyph');

figure; showRectifiedWithEpipolar(J1, J2, 14);

A = imfuse(J1, J2, 'ColorChannels', [1 2 0], 'Scaling', 'joint');
figure; drawEpipolarOnSingle(A, 14);

%% Save rectified RGB and masks
imwrite(J1,  'Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\Ls4f35_rect_m.png');
imwrite(J2,  'Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\Rs4f35_rect_m.png');
imwrite(J1m, 'Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\Ls4f35_mask_rect.png');
imwrite(J2m, 'Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\Rs4f35_mask_rect.png');

%% Export rectified meta (fx, baseline_mm)
fx = Q(3,4);                 % verify once with a known-depth target
b_m = 1 / Q(4,3);            % baseline in meters
baseline_mm = abs(b_m) * 1000;   % IMPORTANT: convert to mm

orig_width = size(I1, 2);
rect_width = size(J1, 2);

save('Floor crack ICT corriedor\Slab test\all_crack\site\s4f35\rect_meta_datas4f35.mat', ...
     'fx', 'baseline_mm', 'orig_width', 'rect_width');
