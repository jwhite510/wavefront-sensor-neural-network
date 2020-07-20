clear all
clc

[diffraction_pattern_file, support_file, retrieved_obj_file, reconstructed_file, real_interp_file, imag_interp_file] = loaddata();

diffraction = load(diffraction_pattern_file, 'diffraction');
diffraction = getfield(diffraction, 'diffraction');
support = load(support_file, 'support');
support = getfield(support, 'support');

% diffraction = load('diffraction.mat', 'diffraction');
% diffraction = getfield(diffraction, 'diffraction');
% support = load('support.mat', 'support');
% support = getfield(support, 'support');

meas_diff = double(diffraction);
seed_obj = double(support);

% the parameters
beta = 0.95; %between 0.8 and 0.98; if higher, feedback is stronger
iter_total = 800;  % total number of iterations
iter_cycle = 800;   % cycle of HIO/RAAR and ER
HIO_num = 800;      % HIO/RAAR per cycle
parameters = [beta iter_total iter_cycle HIO_num];

in_BS = NaN;
BS_used = 0 ;
update_me = 0 ;         % get an GUI update or not, slow if updated
use_RAAR = 1;           % 1 for RAAR and 0 for HIO
conditionals = [BS_used update_me use_RAAR];

% run this for the same diffraction patterns used in the neural network retrieval
% tic
[rec_object, ref_support, err_fourier_space, err_obj_space, recon_diffracted] = seeded_reconst_func(meas_diff, parameters, conditionals, in_BS, seed_obj);
% rec_object(1:64,:)=0.1*max(max(abs(rec_object)))+0.1*max(max(abs(rec_object)))*1i;
% toc
reconst_plot=1;
reconst_interp=1;
num_pxls_dp = 512;            % number of pixels of the detector
cam_pxl_size = 13.5;
samp_cam_dist=32000;
lam = 0.0135;       % in microns
on_chip_binning=2;
phase_threshold = 0.1;
%% ploting the results
% -------------------------------------------------------------------
if reconst_plot == 1
    % Generate grid of spatial frequencies
    theta_max_edge = atand((0.5*num_pxls_dp*cam_pxl_size)/samp_cam_dist); 
    q_max_measured = 2*sind(theta_max_edge/2)/lam;
    
    q_x_cam_plus = linspace(0,q_max_measured,0.5*size(meas_diff,1));
    q_x_cam_minus = -fliplr(q_x_cam_plus(2:end));
    q_x_cam_minus = [2*q_x_cam_minus(1)-q_x_cam_minus(2) q_x_cam_minus];

    q_x_cam = [q_x_cam_minus q_x_cam_plus];
    
    q_x_cam_cr = q_x_cam;
    
    num_pxls = (num_pxls_dp)/on_chip_binning;
    
    q_x_cam_arr = downsample(q_x_cam_cr, 1);
    q_y_cam_arr = q_x_cam_arr;
    [q_x_cam,q_y_cam] = meshgrid(q_x_cam_arr ,q_y_cam_arr);

    % df = (1 / N * dx)
    % dx = (1 / N * df)
    % calcute frequency axes from position axes
    dfgh=size(q_x_cam);
    N=dfgh(1);
    d_alpha=q_x_cam_arr(2)-q_x_cam_arr(1);
    d_x = (1 / N * d_alpha);
    % make linspace
    x_sam1=-N/2:1:N/2-1;
    x_sam1=x_sam1*d_x;
    [x_sam,y_sam] = meshgrid(x_sam1,x_sam1);
    % [AC, x_sam, y_sam] = fourier2Dplus(meas_diff, q_x_cam, q_y_cam);    % just to get the axes 



    % x_sam=()

    % fig1 = classFig('PPT');
    % fig = imagesc(x_sam(1,:), y_sam(:,1),(abs(rec_object))); 
    title('Reconstructed amplitude')
    title_val2 = ['Reconstructed amp, beta = ',num2str(beta)];
    title(title_val2)
    colormap hot
    colorbar
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Amp.png'), 'png')


    rec_object3 = rec_object;
    [m, ind] = max(abs(rec_object3(:)));
    [x, y] = ind2sub(size(rec_object3), ind);
    abs(rec_object3(x, y))
    rec_object3(abs(rec_object3)<0.05 * max(max(abs(rec_object3)))) = NaN;
    rec_phs = angle(rec_object3 * exp(-1j*angle(rec_object3(x,y))));
    
    norm_phase = angle(rec_object3);
    norm_phase(isnan(norm_phase)) = 0;
    range = max(max(norm_phase)) - min(min(norm_phase));

    % fig2 = classFig('PPT');
    % fig = imagesc(x_sam(1,:), y_sam(:,1), rec_phs); 
    colormap hot
    colorbar
    title_val = ['Reconstructed phase, beta = ',num2str(beta)];
    title(title_val)
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Phase.png'))

    % 
    % fig3 = classFig('PPT');
    % fig = imagesc(log10(meas_diff));
    caxis([0 5])
    colorbar
    title('Measured diffraction pattern')
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Measure_Diff.png'))

    % 
    % % fig4 = classFig('PPT');
    % fig = imagesc(log10(abs(recon_diffracted)));
    caxis([0 5])
    colorbar
    title('Reconstructed diffraction pattern')
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Reconst_Diff.png'))
    
    figure
    fig = imagesc(ref_support);
    title('Final support')
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_final support.png'))
    
    % 
    % fig7 = classFig('PPT');
    fig = plot(err_obj_space);
    hold on
    plot(err_fourier_space, 'Color','r')
    legend('Error obj space', 'Error Fourier space')
    title('Error metrics - real and Fourier space')
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Error_metrics.png'))

    % fig8 = classFig('PPT');
    % fig = imagesc(x_sam(1,:), y_sam(:,1),(imag(rec_object))); 
    title('Imagenary part')
    title_val2 = ['Imaginary part obj., beta = ',num2str(beta)];
    title(title_val2)
    colormap hot
    colorbar
    % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_Imag_part_recon.png'))
    


end

%% interpolation
seed_obj = ref_support;
if reconst_interp == 1
    
    rec_object(isnan(rec_object)) = 0;
    rec_phs(isnan(rec_phs)) = 0;
    rec_object(seed_obj==0) = 0;
    
    CC = bwconncomp(abs(rec_object)); 
    
    regions = CC.PixelIdxList;
    obj_ave_real = NaN(size(meas_diff, 2));
    obj_ave_imag = NaN(size(meas_diff, 2));

    obj_amp = abs(rec_object3);
    obj_amp = obj_amp/max(max(obj_amp));
    obj_amp(isnan(obj_amp)) = 0;
    obj_amp = imgaussfilt(obj_amp, 5);
    obj_amp(seed_obj==0) = 0;
    
    % figure
    % imagesc(obj_amp)
    
    norm_phase(obj_amp < phase_threshold) = 0;
    norm_phase = rec_phs;

    % use the real and imaginary part to do interpolation
    obj_real = real(rec_object3);
    obj_real = obj_real/max(max(obj_real));
    obj_real(isnan(obj_real)) = 0;
    obj_real = imgaussfilt(obj_real, 5);
    obj_real(seed_obj==0) = 0;

    obj_imag = imag(rec_object3);
    obj_imag = obj_imag/max(max(obj_imag));
    obj_imag(isnan(obj_imag)) = 0;
    obj_imag = imgaussfilt(obj_imag, 5);
    obj_imag(seed_obj==0) = 0;
    
        for i=1:length(regions)
            IND = cell2mat(regions(i));
            s = [size(meas_diff,1),size(meas_diff,2)];
            [Ind_row,Ind_col] = ind2sub(s,IND);
            amp_ave = 0;
            ph_ave = 0;
            
            temp_amps = zeros(size(IND));
            reals = zeros(size(IND));
            imags = zeros(size(IND));
            
            for j=1:length(IND)
                temp_amps(j) = abs(rec_object(Ind_row(j),Ind_col(j)));
                     %amp_ave = amp_ave + abs(rec_object(Ind_row(j),Ind_col(j)));
                     %ph_ave = ph_ave + norm_phase(Ind_row(j),Ind_col(j));
            end
            
            maximum = max(temp_amps);
            for j=1:length(IND)
                if temp_amps(j) > 0.5*maximum
                    imags(j) = obj_imag(Ind_row(j),Ind_col(j));
                end
                if temp_amps(j) > 0.3*maximum
                         %reals(j) = abs(rec_object(Ind_row(j),Ind_col(j)));
                     reals(j) = obj_real(Ind_row(j),Ind_col(j));
                end
            end
            
            imags = imags(imags~=0);
            reals = reals(reals~=0);
            
            Ind_row_ave = ceil(sum(Ind_row)/length(Ind_row));
            Ind_col_ave = ceil(sum(Ind_col)/length(Ind_col));
            
                 %obj_ave_real(Ind_row_ave,Ind_col_ave) = amp_ave/length(IND);
                 %obj_ave_imag(Ind_row_ave,Ind_col_ave) = ph_ave/length(IND);

            obj_ave_real(Ind_row_ave,Ind_col_ave) = mean(reals);
            obj_ave_imag(Ind_row_ave,Ind_col_ave) = mean(imags);

        end

             %fig8 =classFig('PPT2');
             %imagesc(x_sam(1,:), y_sam(:,1),obj_ave_real); 
             %colormap hot 
             %fig4 =classFig('PPT2');
             %imagesc(x_sam(1,:), y_sam(:,1),obj_ave_imag); 

        % plot the real and imaginary retrieved
        close all;
        figure();
        h=pcolor(real(rec_object));
        title("real(rec object)");
        set(h,'EdgeColor','none');
        figure();
        h=pcolor(imag(rec_object));
        title("imag(rec object)");
        set(h,'EdgeColor','none');

        valid = ~isnan(obj_ave_real);
        obj_real_interp1 = griddata(x_sam(valid),y_sam(valid),obj_ave_real(valid),x_sam,y_sam,'cubic');
        
        % fig9 =classFig('PPT');
        % figure();
        % fig = imagesc(x_sam(1,:), y_sam(:,1),obj_real_interp1); 
        % title('real interpolation')
        % colormap hot
        % colorbar
        % % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_amp_interpol.png'))

        figure();
        h=pcolor(obj_real_interp1);
        title("obj real interp1");
        set(h,'EdgeColor','none');

        valid = ~isnan(obj_ave_imag);
        obj_imag_interp1 = griddata(x_sam(valid),y_sam(valid),obj_ave_imag(valid),x_sam,y_sam,'cubic');
        
        % % fig5 = classFig('PPT');
        % figure();
        % fig = imagesc(x_sam(1,:), y_sam(:,1),obj_imag_interp1);
        % title('imag interpolation')
        % colorbar
        % % saveas(fig, strcat(dump_folder, '\\', int2str(samp_cam_dist), '_phs_interpol.png'))

        figure();
        h=pcolor(obj_imag_interp1);
        title("obj imag interp1");
        set(h,'EdgeColor','none');

end

% figure
% imagesc(abs(rec_object))
% title('abs(rec_object)')
% saveas(gcf, 'abs_rec_object.png')

% figure
% imagesc(real(rec_object))
% title('real(rec_object)')
% saveas(gcf, 'real_rec_object.png')

% figure
% imagesc(imag(rec_object))
% title('imag(rec_object)')
% saveas(gcf, 'imag_rec_object.png')

% save .mat file
save(retrieved_obj_file, 'rec_object');
save(reconstructed_file, 'recon_diffracted');
save(real_interp_file, 'obj_real_interp1');
save(imag_interp_file, 'obj_imag_interp1');
exit();



