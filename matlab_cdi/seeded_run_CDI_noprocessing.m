clear all
clc

[diffraction_pattern_file, support_file, retrieved_obj_file, reconstructed_file] = loaddata();

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
beta = 0.9; %between 0.8 and 0.98; if higher, feedback is stronger
iter_total = 200;  % total number of iterations
iter_cycle = 201;   % cycle of HIO/RAAR and ER
HIO_num = 100;      % HIO/RAAR per cycle
parameters = [beta iter_total iter_cycle HIO_num];

in_BS = NaN;
BS_used = 0 ;
update_me = 0 ;         % get an GUI update or not, slow if updated
use_RAAR = 1;           % 1 for RAAR and 0 for HIO
conditionals = [BS_used update_me use_RAAR];

% run this for the same diffraction patterns used in the neural network retrieval
[rec_object, ref_support, err_fourier_space, err_obj_space, recon_diffracted] = seeded_reconst_func(meas_diff, parameters, conditionals, in_BS, seed_obj);

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
exit();



