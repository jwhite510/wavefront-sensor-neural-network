% This is a script for retrieving the phase of a diffraction pattern thus enabling the 
% reconstruction of the diffracting object in real space.
% Date : 23.11.2018
% important parameters : samp_cam_dist, sub_image, on_chip_binning, crop, sam_size
% important points : stray light, threshold for background, scaling of
% support, flip/not of the support, correct sample size
% Mask the center...

close all
clear all
clc

addpath('Y:\Software\Matlab_classFig'); %path to class_fig

% If a simulation should be run
sim = false;

%% loading the data
if ~sim
    % Loading measurement data
%     [name,pathstr,~] = uigetfile('Y:\AG_HHG\experiments\2018_11_Wavefront_sensing\20200127\4um_pinhole\z-360\feld_2\fits\*.fits');
%     [name,pathstr,~] = uigetfile('Y:\AG_HHG\experiments\2018_11_Wavefront_sensing\20200127\4um_pinhole\z0\feld_2\fits\*.fits');
%     [name,pathstr,~] = uigetfile('Y:\AG_HHG\experiments\2018_11_Wavefront_sensing\20200221\wfs_3um_pinhole\fits\*.fits');
    [name,pathstr,~] = uigetfile('Y:\AG_HHG\experiments\2018_11_Wavefront_sensing\20200224\feld_2\z-1000\fits\*.fits');
    file=fullfile(pathstr,name);
    data = fitsread(file);
else
    % % loading simulation data
    load('\20200108\fits\FELD3_0p5s_2x2b.fits');
    data = DP;
end

% support_filename = 'samples\\size_3p2um_pitch_400nm_diameter_200nm_psize_5nm.png';
support_filename = 'samples\\size_6um_pitch_600nm_diameter_300nm_psize_5nm.png';
supp_pxl_size = 0.005;          % psize in the png image [um/px]

meas_diff = double(data);

%% Output settings
reconst_interp = 1;     % 1 to do interpolation of the reconstruction
reconst_plot = 1;       % 1 to plot the reconstructions
dump_recon = true;    % If the reconstruction should be saved in a mat file
dump_folder = 'test';
phase_threshold = 0.1;
% Create dump folder if it does not exists
if ~exist(dump_folder, 'dir')
    mkdir(dump_folder)
end

%% settings of the experiment
% do some rework here and only use necessary parameters
lam = 0.0135;       % in microns
% samp_cam_dist_l = [32500, 32501, 32502, 32750, 32751, 33000, 33001, 33002];
samp_cam_dist_l = [32000, 32001, 32250, 32251];
cam_pxl_size = 13.5;
on_chip_binning = 2; % during measurement. camera binning to get the loaded data
binning = 1;         % additional numerical binning to be applied now      
crop = 0;           % 0 for no additional cropping, 1 for half the camera and 2 for 1/4 of the camera
pad = false;
total_bin = binning * on_chip_binning;
num_pxls_dp = 512;            % number of pixels of the detector
window = true;

if pad > num_pxls_dp
    num_pxls_dp = pad;
end
    
%% Settings of the reconsruction algorithm
% ----------------------------------------------------------------------
stiching = false;
in_BS = NaN;
BS_used = 0 ;
update_me = 0 ;         % get an GUI update or not, slow if updated
use_RAAR = 1;           % 1 for RAAR and 0 for HIO
conditionals = [BS_used update_me use_RAAR];

beta = 0.8; %between 0.8 and 0.98; if higher, feedback is stronger
iter_total = 49;  % total number of iterations
iter_cycle = 50;   % cycle of HIO/RAAR and ER
HIO_num = 40;      % HIO/RAAR per cycle
parameters = [beta iter_total iter_cycle HIO_num];

%% Preprocessing parameters
%-------------------------------------------------------------------------
cc = true;
threshold = 20;
offset = 0;
% rot_angle = -1.25;  % First data set
rot_angle = 2.7;
long_exp_t = 10;
short_exp_t = 1;
sigma_conv = 0.1;     % 0.1

%% Preprocessing
% Stiching
if stiching
    factor = long_exp_t / short_exp_t;
    
    % Load short exposure data
    [name,pathstr,~] = uigetfile('Y:\AG_HHG\experiments\2018_11_Wavefront_sensing\20181212\fits\*.fits');
    file = fullfile(pathstr,name);
    short_exp = fitsread(file);
    short_exp = double(short_exp);
    
    short_exp(short_exp < 5) = 0;
    
    figure
    imagesc(meas_diff)
    caxis([30000 50000])
    colorbar
    title('Measured diffraction pattern')
    
    load('mask.mat');
    
    figure
    imagesc(mask)
    title('loaded mask')
    
    figure
    imagesc(meas_diff .* mask, [0, 1000])
    title('part one')
    
    figure
    imagesc((mask-1)*(-1) .* short_exp * factor, [0, 1000])
    title('part 2')
    
    meas_diff = meas_diff .* mask + (mask-1)*(-1) .* short_exp * factor;
    
end

% Centering
[diff_max, diff_max_ind] = max(meas_diff);
[~, diff_ind_c] = max(diff_max);
diff_ind_r = diff_max_ind(diff_ind_c);

raw_diff_int = imtranslate(meas_diff, [(0.5*size(meas_diff,1)-diff_ind_c)+1, (0.5*size(meas_diff,2)-diff_ind_r)+1]);

if crop
    num_pxl = size(raw_diff_int, 1);
    raw_diff_int = raw_diff_int(num_pxl / 2 - num_pxl /4 + 1: num_pxl / 2 + num_pxl /4, num_pxl / 2 - num_pxl /4 + 1: num_pxl / 2 + num_pxl /4);
    num_pxls_dp = size(raw_diff_int, 1);
end  

diff_int = databin(raw_diff_int, binning);  % .*1e3;

% rotate the diffraction pattern
if ~sim
    diff_int = imrotate(diff_int,rot_angle,'crop');
end
temp_2 = diff_int;

for i = 1:size(samp_cam_dist_l, 2)
    diff_int = temp_2;
    samp_cam_dist = samp_cam_dist_l(i);
    obj_pxl_size = lam / (num_pxls_dp * cam_pxl_size * total_bin / samp_cam_dist);    % resolution in the object plane
    
    if ~sim && cc
        % curvature correction
        diff_int = curvature_correction(temp_2, cam_pxl_size*total_bin*1e-6, samp_cam_dist*1e-6, lam*1e-6);
        diff_int(isnan(diff_int)) = 0.;
    end
    
    if isa(pad, 'double')
        add_y = (pad - size(diff_int, 1)) / 2.;
        add_x = (pad - size(diff_int, 2)) / 2.;
        diff_int = padarray(diff_int, [add_y, add_x], 'both');
    end
    
    figure
    imagesc(diff_int)
    
    % Setting the offset
    diff_int = diff_int - offset;
    
    % Thresholding of the noise
    diff_int(diff_int<threshold) = 0;
    
    if window
       wind = hanning(size(diff_int, 1)) * hanning(size(diff_int, 2))';
       diff_int = diff_int .* wind;
    end
    
    figure
    imagesc(diff_int, [0, 200])
    title('Diff int. after preproc')

    %% seeding the sample layout as a support
    bmp_img = imread(support_filename);
    bmp_gray = double(rgb2gray(bmp_img));
    
    bmp_gray = bmp_gray ./ max(max(bmp_gray));
    bmp_gray = (bmp_gray - 1) .* -1;
    bmp_gray = imgaussfilt(bmp_gray, sigma_conv);
    
    FOV = (lam*samp_cam_dist)/(cam_pxl_size*total_bin);  % I don't know where the 2 in the denominator came from

    seed_obj_downsamp = imresize(bmp_gray, supp_pxl_size/obj_pxl_size);
    
    seed_obj_downsamp = double(seed_obj_downsamp>0.03);   % to make it binary
    pad_size = floor((num_pxls_dp - size(seed_obj_downsamp, 1))/2.);
    seed_obj = padarray(seed_obj_downsamp, [pad_size, pad_size], 0, 'both');
    
    if size(seed_obj) ~= size(diff_int)
        temp = zeros(size(diff_int));
        temp(1: end - 1, 1: end - 1) = seed_obj;
        seed_obj = temp;
        size(seed_obj)
    end
    
    seed_obj = fliplr(seed_obj);
%     seed_obj = flipud(seed_obj); % correct??

    figure
    fig = imagesc(seed_obj);
    title('seeded object adjusted')
    saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_seeded support.png'))

    %% calling the reconstruction function
    % ------------------------------------------------------------------
    [rec_object, ref_support, err_fourier_space, err_obj_space, recon_diffracted] = seeded_reconst_func(diff_int, parameters, conditionals, in_BS, seed_obj);

    %% ploting the results
    % -------------------------------------------------------------------
    if reconst_plot == 1
        % Generate grid of spatial frequencies
        theta_max_edge = atand((0.5*num_pxls_dp*cam_pxl_size)/samp_cam_dist); 
        q_max_measured = 2*sind(theta_max_edge/2)/lam;
        
        q_x_cam_plus = linspace(0,q_max_measured,0.5*size(diff_int,1));
        q_x_cam_minus = -fliplr(q_x_cam_plus(2:end));
        q_x_cam_minus = [2*q_x_cam_minus(1)-q_x_cam_minus(2) q_x_cam_minus];

        q_x_cam = [q_x_cam_minus q_x_cam_plus];
        
        q_x_cam_cr = q_x_cam;
        
        num_pxls = (num_pxls_dp)/on_chip_binning;
        
        q_x_cam_arr = downsample(q_x_cam_cr, 1);
        q_y_cam_arr = q_x_cam_arr;
        
        [q_x_cam,q_y_cam] = meshgrid(q_x_cam_arr ,q_y_cam_arr);
        [AC, x_sam, y_sam] = fourier2Dplus(diff_int, q_x_cam, q_y_cam);    % just to get the axes 
        
        fig1 = classFig('PPT');
        fig = imagesc(x_sam(1,:), y_sam(:,1),(abs(rec_object))); 
        title('Reconstructed amplitude')
        title_val2 = ['Reconstructed amp, beta = ',num2str(beta)];
        title(title_val2)
        colormap hot
        colorbar
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Amp.png'), 'png')


        rec_object3 = rec_object;
        [m, ind] = max(abs(rec_object3(:)));
        [x, y] = ind2sub(size(rec_object3), ind);
        abs(rec_object3(x, y))
        rec_object3(abs(rec_object3)<0.05 * max(max(abs(rec_object3)))) = NaN;
        rec_phs = angle(rec_object3 * exp(-1j*angle(rec_object3(x,y))));
        
        norm_phase = angle(rec_object3);
        norm_phase(isnan(norm_phase)) = 0;
        range = max(max(norm_phase)) - min(min(norm_phase));

        fig2 = classFig('PPT');
        fig = imagesc(x_sam(1,:), y_sam(:,1), rec_phs); 
        colormap hot
        colorbar
        title_val = ['Reconstructed phase, beta = ',num2str(beta)];
        title(title_val)
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Phase.png'))

        % 
        fig3 = classFig('PPT');
        fig = imagesc(log10(diff_int));
        caxis([0 5])
        colorbar
        title('Measured diffraction pattern')
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Measure_Diff.png'))

        % 
        fig4 = classFig('PPT');
        fig = imagesc(log10(abs(recon_diffracted)));
        caxis([0 5])
        colorbar
        title('Reconstructed diffraction pattern')
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Diff.png'))
        
        figure
        fig = imagesc(ref_support);
        title('Final support')
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_final support.png'))
        
        % 
        fig7 = classFig('PPT');
        fig = plot(err_obj_space);
        hold on
        plot(err_fourier_space, 'Color','r')
        legend('Error obj space', 'Error Fourier space')
        title('Error metrics - real and Fourier space')
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Error_metrics.png'))

        fig8 = classFig('PPT');
        fig = imagesc(x_sam(1,:), y_sam(:,1),(imag(rec_object))); 
        title('Imagenary part')
        title_val2 = ['Imaginary part obj., beta = ',num2str(beta)];
        title(title_val2)
        colormap hot
        colorbar
        saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Imag_part_recon.png'))
        


    end
    
    %% interpolation
    seed_obj = ref_support;
    if reconst_interp == 1
        
        rec_object(isnan(rec_object)) = 0;
        rec_phs(isnan(rec_phs)) = 0;
        rec_object(seed_obj==0) = 0;
        
        CC = bwconncomp(abs(rec_object)); 
        
        regions = CC.PixelIdxList;
        obj_ave_amp = NaN(size(diff_int, 2));
        obj_ave_ph = NaN(size(diff_int, 2));

        obj_amp = abs(rec_object3);
        obj_amp = obj_amp/max(max(obj_amp));
        obj_amp(isnan(obj_amp)) = 0;
        obj_amp = imgaussfilt(obj_amp, 5);
        obj_amp(seed_obj==0) = 0;
        
        figure
        imagesc(obj_amp)
        
        norm_phase(obj_amp < phase_threshold) = 0;
        norm_phase = rec_phs;
        
            for i=1:length(regions)
                IND = cell2mat(regions(i));
                s = [size(diff_int,1),size(diff_int,2)];
                [Ind_row,Ind_col] = ind2sub(s,IND);
                amp_ave = 0;
                ph_ave = 0;
                
                temp_amps = zeros(size(IND));
                amps = zeros(size(IND));
                phs = zeros(size(IND));
                
                for j=1:length(IND)
                    temp_amps(j) = abs(rec_object(Ind_row(j),Ind_col(j)));
%                     amp_ave = amp_ave + abs(rec_object(Ind_row(j),Ind_col(j)));
%                     ph_ave = ph_ave + norm_phase(Ind_row(j),Ind_col(j));
                end
                
                maximum = max(temp_amps);
                for j=1:length(IND)
                    if temp_amps(j) > 0.5*maximum
                        phs(j) = norm_phase(Ind_row(j),Ind_col(j));
                    end
                    if temp_amps(j) > 0.3*maximum
%                         amps(j) = abs(rec_object(Ind_row(j),Ind_col(j)));
                         amps(j) = obj_amp(Ind_row(j),Ind_col(j));
                    end
                end
                
                phs = phs(phs~=0);
                amps = amps(amps~=0);
                
                Ind_row_ave = ceil(sum(Ind_row)/length(Ind_row));
                Ind_col_ave = ceil(sum(Ind_col)/length(Ind_col));
                
%                 obj_ave_amp(Ind_row_ave,Ind_col_ave) = amp_ave/length(IND);
%                 obj_ave_ph(Ind_row_ave,Ind_col_ave) = ph_ave/length(IND);

                obj_ave_amp(Ind_row_ave,Ind_col_ave) = mean(amps);
                obj_ave_ph(Ind_row_ave,Ind_col_ave) = mean(phs);

            end

% 
%             fig8 =classFig('PPT2');
%             imagesc(x_sam(1,:), y_sam(:,1),obj_ave_amp); 
%             colormap hot 
%             fig4 =classFig('PPT2');
%             imagesc(x_sam(1,:), y_sam(:,1),obj_ave_ph); 

            valid = ~isnan(obj_ave_amp);
            obj_amp_interp1 = griddata(x_sam(valid),y_sam(valid),obj_ave_amp(valid),x_sam,y_sam,'cubic');
            
            fig9 =classFig('PPT');
            fig = imagesc(x_sam(1,:), y_sam(:,1),obj_amp_interp1); 
            title('amp interpolation')
            colormap hot
            colorbar
            saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_amp_interpol.png'))

            valid = ~isnan(obj_ave_ph);
            obj_ph_interp1 = griddata(x_sam(valid),y_sam(valid),obj_ave_ph(valid),x_sam,y_sam,'cubic');
            
            fig5 = classFig('PPT');
            fig = imagesc(x_sam(1,:), y_sam(:,1),obj_ph_interp1);
            title('phs interpolation')
            colorbar
            saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_phs_interpol.png'))
    end
    if dump_recon
        close all  
        % Save workspace
        save(strcat(dump_folder, '\\', int2str(samp_cam_dist),'_recon_results'));
    end
end
