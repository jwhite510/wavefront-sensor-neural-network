function [final_object, support, err_fourier_space, err_obj_space, recon_diffracted] = seeded_reconst_func(diff_int, parameters, conditionals, in_BS, seed_obj)

BS_used = conditionals(1) ;
update_me = conditionals(2) ;
use_RAAR = conditionals(3) ;
positivity = true;

beta = parameters(1);
iter_total = parameters(2);
iter_cycle = parameters(3);
HIO_num = parameters(4);

pixel_nr_y = size(diff_int,1);
pixel_nr_x = size(diff_int,2);

phase_rand = -pi + 2*pi*rand(pixel_nr_y,pixel_nr_x);
full_diff_field = diff_int.^0.5 .* exp(1i*phase_rand);
diff_amp = abs(full_diff_field);

in_support = double(seed_obj);
out_of_support = double(not(seed_obj));

Err = zeros(1,iter_total);
Err2 = zeros(1,iter_total);
fourier_error = zeros(1,iter_total);

P_filtered = full_diff_field;
if update_me == 1
figure;
set(gcf,'Units', 'Normalized', 'OuterPosition', [0.05 0.3 0.9 0.7]); 
end

reduce_supp = zeros(length(diff_int));
% iteration starts
for i=1:iter_total
    % i
    if mod(i,100) == 0
        i
    end
    S = ifftshift(ifft2(ifftshift(P_filtered)));    % to real space 
    
    sum_total = sum(sum(abs(S).^2));
    sum_out = sum(sum(abs(S.*out_of_support).^2));
    sum_in = sum(sum(abs(S.*in_support).^2));
    Err(1,i) = sqrt(sum_out/sum_total);                   % error in real space
    Err2(1,i) = sqrt(sum_out/sum_in);                     % alternative formulation

        count = rem(i,iter_cycle);    
        in_support_prev = in_support;
        
% %         reduce_supp(:,1:0.5*length(reduce_supp)) = 1;
%         reduce_supp(:,0.5*length(reduce_supp):end) = 1;
% 
%        if i > 500 && i < 550  
%            in_support = reduce_supp.*in_support;    % support reduction for escaping from twin-image problem
%            out_of_support = double(not(logical(in_support)));
%        else   
%            in_support = double(seed_obj);
%            out_of_support = double(not(seed_obj));
%        end
       
    %% object space constraint
% ---------------------------------------------------------------------------------------        
      if count<HIO_num %|| i>0.75*iter_total
         % HIO or RAAR

         % Hybrid input-output algorithm or RAAR
         if i==1 
             S_filtered = S;
         else if i~=1 && use_RAAR==0
             obj_in_supp = S.*in_support;       % common version: S, original version: S_prev
             obj_out_supp = (S_prev - beta*S).*out_of_support;
             S_filtered = obj_in_supp + obj_out_supp;
             else 
             beta_n = beta;     % + (1 - beta)*(1 - exp(-(0.3*iter_total/7)^3));                  
             obj_in_supp = S.*in_support;
             obj_out_supp = (beta_n*S_prev + (1 - 2*beta_n)*S).*out_of_support;
             S_filtered = obj_in_supp + obj_out_supp;       
             end
         end

         S_prev = S_filtered;
         
      else 
        % Error reduction 
        in_support = double(seed_obj);
        S_filtered = S.*in_support; 
        S_prev = S_filtered;
      end

     
%% Fourier Space operations
% -------------------------------------------------------------------------------------
    P = fftshift(fft2(fftshift(S_filtered)));       % to Fourier space
       
    fourier_error(1,i) = sqrt(sum(sum((abs(P) - diff_amp).^2))/sum(sum(diff_int)));
    
    if BS_used == 1
%        diff_amp_BS = diff_amp + abs(P.*in_BS);
       diff_amp_BS = diff_amp.*double(not(logical(in_BS))) + abs(P.*in_BS);
%        diff_amp_BS = diff_amp.*double(not(logical(in_BS))) + max(abs(P),diff_amp).*in_BS;    % value is replaced only if it is greater than the measured one
       P_filtered = diff_amp_BS .*exp(1i*angle(P));       % Fourier constraint I
    else
       P_filtered = diff_amp .*exp(1i*angle(P));          % Fourier constraint I
    end
    
    % Apply positivity constraint
%     if positivity && i > 200
%     S(imag(S)<0) = 0;
%     end
%     
%     if mod(i, 1000) == 0        
%         figure
%         imagesc(abs(S))
%         title('Amplitude after pos constraint.')
%         colormap hot
%         colorbar
%         
%         h = figure
%         imagesc(imag(S))
%         title('Imaginary part after pos. constraint.')
%         colormap hot
%         colorbar
%         drawnow;
%         
%         figure
%         temp = S;
%         temp(abs(temp)<0.03*max(max(abs(temp)))) = NaN;
%         imagesc(angle(temp)); 
%         title('phase')
%         colormap hot
%         colorbar
% 
%         waitfor(h)
%     end
    
%% for updating the figures
% -------------------------------------------------------------------------------------
    
    if rem(i,50) == 0 && update_me ==1
        subplot(1,2,1); 
%        pcolor(x_vec,y_vec,rot90(abs(S),2)); shading interp
       pcolor(abs(S)); shading interp
       set(gca,'FontName','Calibri','FontSize',18);
       axis square
       hold on 
       drawnow
       
       % to save animation into a movie
%        frame = getframe;               % uncomment for animation purposes
%        writeVideo(v,frame);            % uncomment for animation purposes
       
       subplot(1,2,2);
       scatter(i,fourier_error(i));
       axis square
       hold on
       drawnow 
            
    end
end

%close(v);              % uncomment for animation purposes
final_object = S;
support = in_support_prev;
err_fourier_space = fourier_error;
err_obj_space = Err;
recon_diffracted = abs(P).^2;
