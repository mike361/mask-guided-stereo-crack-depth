%% rectify_stereo_pairs.m
% Purpose:
%   Rectify stereo RGB images and their predicted crack masks using MATLAB stereoParams.

% Inputs:
%   - left_img_path, right_img_path: paths to original left/right RGB images
%   - left_mask_path, right_mask_path: paths to predicted left/right masks (PNG)
%   - calib_mat_path: path to calibrationSession.mat (stereoParams)
%   - out_dir: output directory for rectified images/masks/meta
%   - outputView: 'valid' (default) or 'full'

% Outputs (saved to out_dir):
%   - <left_base>_rect.png, <right_base>_rect.png
%   - <left_base>_mask_rect.png, <right_base>_mask_rect.png
%   - rect_meta_data.mat (fx, baseline_mm, orig_width, rect_width)

% Notes:
%   - OutputView='valid' enforces a common valid region after rectification.
%   - Mask interpolation MUST be nearest-neighbor to preserve binary labels.

% How to run:
%rectify_stereo_pairs( ...
%  "data/left/Ls4f35_m.JPG", ...
%  "data/right/Rs4f35_m.JPG", ...
%  "outputs/masks/pred_mask_Ls4f35.png", ...
%  "outputs/masks/pred_mask_Rs4f35.png", ...
%  "calib/calibrationSession.mat", ...
%  "outputs/rectified", ...
%  "valid");


function rectify_stereo_pairs(left_img_path, right_img_path, left_mask_path, right_mask_path, calib_mat_path, out_dir, outputView)
%RECTIFY_STEREO_PAIRS Rectify stereo RGB images and predicted masks using stereoParams.
%
% Inputs:
%   left_img_path   - path to left RGB image
%   right_img_path  - path to right RGB image
%   left_mask_path  - path to left predicted mask (png)
%   right_mask_path - path to right predicted mask (png)
%   calib_mat_path  - path to calibrationSession.mat (contains calibrationSession.CameraParameters)
%   out_dir         - output folder to save rectified images/masks/meta
%   outputView      - 'valid' (default) or 'full'
%
% Outputs (saved to out_dir):
%   left_rect.png, right_rect.png
%   left_mask_rect.png, right_mask_rect.png
%   rect_meta_data.mat (fx, baseline_mm, orig_width, rect_width)
%
% Notes:
%   - Masks are rectified with nearest-neighbor interpolation to preserve binary labels.

if nargin < 7 || isempty(outputView), outputView = 'valid'; end
if nargin < 6 || isempty(out_dir), out_dir = pwd; end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% -------------------- Load inputs --------------------
I1 = imread(left_img_path);
I2 = imread(right_img_path);

M1 = imread(left_mask_path);
M2 = imread(right_mask_path);

S = load(calib_mat_path);
stereoParams = S.calibrationSession.CameraParameters;

% -------------------- Preprocess masks --------------------
if size(M1,3) > 1, M1 = rgb2gray(M1); end
if size(M2,3) > 1, M2 = rgb2gray(M2); end

M1b = bwareaopen(M1 > 0, 100);
M2b = bwareaopen(M2 > 0, 50);

M1c = bwareafilt(M1b, 1);
M2c = bwareafilt(M2b, 1);

M1c = imclose(M1c, strel('disk', 2));
M2c = imclose(M2c, strel('disk', 2));

M11 = M1c > 0;
M22 = M2c > 0;

% -------------------- Rectify --------------------
[J1, J2, Q] = rectifyStereoImages(I1, I2, stereoParams, 'OutputView', outputView);

[J1m, J2m] = rectifyStereoImages(M11, M22, stereoParams, 'nearest', 'OutputView', outputView);
J1m = uint8(J1m) * 255;
J2m = uint8(J2m) * 255;

% -------------------- Save outputs --------------------
[~, baseL, ~] = fileparts(left_img_path);
[~, baseR, ~] = fileparts(right_img_path);

left_rect_path  = fullfile(out_dir, [baseL '_rect.png']);
right_rect_path = fullfile(out_dir, [baseR '_rect.png']);

left_mask_rect_path  = fullfile(out_dir, [baseL '_mask_rect.png']);
right_mask_rect_path = fullfile(out_dir, [baseR '_mask_rect.png']);

imwrite(J1,  left_rect_path);
imwrite(J2,  right_rect_path);
imwrite(J1m, left_mask_rect_path);
imwrite(J2m, right_mask_rect_path);

% -------------------- Export meta --------------------
fx = Q(3,4);              % focal length in pixels (verify once with known target if needed)
b_m = 1 / Q(4,3);         % baseline in meters
baseline_mm = abs(b_m) * 1000;

orig_width = size(I1, 2);
rect_width = size(J1, 2);

save(fullfile(out_dir, 'rect_meta_data.mat'), ...
    'fx', 'baseline_mm', 'orig_width', 'rect_width');

fprintf('[saved] Rectified images/masks -> %s\n', out_dir);
fprintf('[saved] rect_meta_data.mat\n');
end

