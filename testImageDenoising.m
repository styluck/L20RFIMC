clc;clear;close all;
addpath('libs/PROPACK','utils','solvers',...
    'algorithm','algorithm/Maxide','algorithm/nuclear_active');
addpath('algorithm/sgimc');
filename = '070'; % 10 20 25 070
ImageRaw = imread(['dataset/images/',filename,'.png']);
Mdata = im2double(ImageRaw(:,:,1));

m = size(Mdata,1);
n = size(Mdata,2); 

randstate = 100;
rng(randstate,'twister'); % Set random seed

psnr_index = @(Irec, Itrue) 10 * log10(1 / mean((Irec(:) - Itrue(:)).^2));
%%
iter_num = 500;
SR = 0.2; % 15 10 25 20

Omega_idx = rand(m,n);
Omega = Omega_idx>1 - SR;
Omega_c = ones(m,n) - Omega;
Obs = Mdata.*Omega;

mu1 = 0.5; mu2 = 0.01;
delta = 0.0;%.1; % noise ratio
r = 50;
[A, s, B] =lansvd(Mdata,r);
A_est = A;
B_est=B;
Noise = randn(n,r);
B_est = B_est+delta*norm(B_est,'fro')/norm(Noise,'fro')*Noise;
PV = pinv(B_est')*B_est';
PVc =eye(n) - pinv(B_est')*B_est';

nr = m; d1 = m;
nc = n; d2 = n;
normM = norm(Mdata, 'fro');
ABnorm = (norm(A_est, 2) * norm(B_est, 2))^2;
bb = Mdata(Omega);

%%
p = length(bb);
randnvec = randn(p,1);
sigma = 0.1 * norm(bb) / norm(randnvec);
xi =  sigma * randnvec;
bb = bb + xi;

%%

normb = norm(bb);
nzidx = find(Omega); 
Mhat = zeros(nr, nc);
Mhat(nzidx) = bb;

pars = struct('normM', normM, ...
    'normb', normb, ...
    'nc', nc, ...
    'nr', nr, ...
    'd1', d1, ...
    'd2', d2, ...
    'p', SR*nr*nc, ...
    'r', r);
%% 
options = 1.0e-6;
OPTIONS_AMM = struct('maxiter', 500, 'printyes', 1, 'tol', 5.0e-5, 'LipX',  ABnorm);
AMB =  A' * Mhat * B;
[U, dA, V] = lansvd(AMB, r, 'L', options);
dA = diag(dA)';
max_dA = max(dA);
Ustart = U(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
Vstart = V(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
tic; % 新增计时

lambda = mu1;% *normb; %.01;%*mu1 * 
L_Fnorm = AMM_Fnorm(Mdata,bb, Ustart, Vstart, ...
            nzidx, A_est, B_est, OPTIONS_AMM, pars, lambda); 
time_Fnorm = toc; % 新增计时变量
psnr_Fnorm = psnr_index(L_Fnorm, Mdata);
err_Fnorm = norm((L_Fnorm-Mdata),'fro')/norm(Mdata,'fro');
fprintf('Fnorm Error = %.3f\n',err_Fnorm);
% 

%%
OPTIONS_SPC = struct('maxiter', 500, 'printyes', 1, ...
    'tol', 1.0e-3, 'Lip_const', .8* ABnorm);

% Pstart = randn(d1, r);
% Qstart = randn(d2, r);
% Pstart = orth(Ustart);%Ustart;%Pstart
% Qstart = orth(Vstart);%Vstart;%Qstart

Pstart = U(:, 1:r);
Qstart = V(:, 1:r);
dstart = ones(1, r);
rat = 2;

% SGIMC parameters
params.lambda_l1 = 0;
params.lambda_group = lambda;
params.lambda_ridge = 0;
params.eta = 1.0;
params.maxOuterIter = 10;
params.maxInnerIter = 10;
params.rtol = 1e-5;
params.atol = 1e-8;
params.loss = 'squared';
params.verbose = true;

tic; % 新增计时
[~, L_AMM] = SInf_AMM_CutCol_new(Mdata, bb, Pstart, Qstart, dstart, ...
                        nzidx, A_est, B_est, lambda, pars, OPTIONS_SPC);
% [Xopt,rankXopt,Loss]= PALM(Ustart,Vstart,nzidx,bb,OPTIONS_AMM,pars,lambda,alpha,beta,Mstar);
time_AMM = toc;
psnr_AMM = psnr_index(L_AMM, Mdata);
 % 新增计时变量
err_AMM = norm((L_AMM-Mdata),'fro')/norm(Mdata,'fro');
fprintf('AMM Error = %.3f\n',err_AMM);

%%

tic; % 新增计时
L_alm_MC = alm_MC(Mhat, Omega, mu1, iter_num,0);
time_MC = toc; % 新增计时变量
err_MC = norm((L_alm_MC-Mdata),'fro')/norm(Mdata,'fro');
psnr_MC = psnr_index(L_alm_MC, Mdata);
fprintf('MC Error = %.3f\n',err_MC);

tic; % 新增计时
lambda = mu1;
[L_Maxide,telapsed_side]=Maxide(Mhat,nzidx,A_est,B_est,lambda,iter_num);
time_Maxide = toc; % 新增计时变量（原elapsed_side可能未使用，此处统一用toc）
err_Maxide = norm((L_Maxide-Mdata),'fro')/norm(Mdata,'fro');
psnr_Maxide = psnr_index(L_Maxide, Mdata);
fprintf('IMC Error = %.3f\n',err_Maxide);

% tic;
% lambda = sqrt(n);
% L_alm_RMC = alm_RMC(Obs, Omega,PVc,lambda,iter_num,mu1,mu2,0);
% time_RMC = toc;
% err = norm((L_alm_RMC-Mdata),'fro')/norm(Mdata,'fro');
% fprintf('RMC Error = %.3f\n',err);

tic; % 新增计时
[Wsg, Hsg, hist_sg] = sgimc_qaadmm(A_est, B_est, Mhat, r, params);
time_SGIMC = toc;
L_SGIMC = (A_est * Wsg) * (B_est * Hsg)';
err_SGIMC = norm(L_SGIMC - Mdata, 'fro') / normM;
psnr_SGIMC = psnr_index(L_SGIMC, Mdata);
fprintf('SGIMC Error = %.3f\n',err_SGIMC);

% L_DirtyIMC = DirtyIMC(Obs,nzidx,A_est,B_est,0.01,1,10,20);
% time_DirtyIMC = toc; % 新增计时变量
% err_DirtyIMC = norm((L_DirtyIMC-Mdata),'fro')/norm(Mdata,'fro');
% fprintf('DirtyIMC Error = %.3f\n',err_DirtyIMC);
  
opts.verbose = 1; 
opts.stop_relRes = 1e-14;   	% small relRes threshold 
opts.stop_relDiff = 1e-14;      % small relative X_hat difference threshold
opts_GNIMC = opts;
opts_GNIMC.alpha = -1;
opts_GNIMC.max_outer_iter = 20;
tic; % 新增计时
[L_GNIMC, ~, ~, ~] = GNIMC(Mhat, nzidx, r, A_est, B_est, opts_GNIMC);
time_GNIMC = toc; % 新增计时变量
err_GNIMC = norm((L_GNIMC-Mdata),'fro')/norm(Mdata,'fro');
psnr_GNIMC = psnr_index(L_GNIMC, Mdata);
fprintf('GNIMC Error = %.3f\n',err_GNIMC);

%% 创建一个新的图形窗口 

% 保存原图像为 PNG
figure('numbertitle','off'); 
imshow(Mdata);
title('Original_Image');
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\Original_Image',filename,'.png']);
close(gcf);

% 保存欠采样图像为 PNG
figure('numbertitle','off'); 
imshow(Obs);
title('Undersampled_Image');
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\Undersampled_Image',filename,'.png']);
close(gcf);

% 保存 Algorithm1 图像为 PNG
figure('numbertitle','off'); 
imshow(L_AMM);
title(sprintf('Algorithm1 (Err=%.3f)', err_AMM));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\Algorithm1',filename,'.png']);
close(gcf);

% 保存 Algorithm2 图像为 PNG
figure('numbertitle','off'); 
imshow(L_Fnorm);
title(sprintf('Algorithm2 (Err=%.3f)', err_Fnorm));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\Algorithm2',filename,'.png']);
close(gcf);

% 保存 IMC 图像为 PNG
figure('numbertitle','off'); 
imshow(L_Maxide);
title(sprintf('IMC (Err=%.3f)', err_Maxide));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\IMC',filename,'.png']);
close(gcf);

% 保存 SGIMC 图像为 PNG
figure('numbertitle','off'); 
imshow(L_SGIMC);
title(sprintf('SGIMC (Err=%.3f)', err_SGIMC));
% imshow(L_DirtyIMC);
% title(sprintf('DirtyIMC (Err=%.3f)', err_DirtyIMC));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\SGIMC',filename,'.png']);
close(gcf);

% 保存 MC 图像为 PNG
figure('numbertitle','off'); 
imshow(L_alm_MC);
title(sprintf('MC (Err=%.3f)', err_MC));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\MC',filename,'.png']);
close(gcf);

% 保存 GNIMC 图像为 PNG
figure('numbertitle','off'); 
imshow(L_GNIMC);
title(sprintf('GNIMC (Err=%.3f)', err_GNIMC));
set (gcf,'units','pixel','Position',[266,400,256,256]);
set(gca, 'position', [0 0 1 1 ]);axis normal;
AFrame=getframe(gcf);
imwrite(AFrame.cdata,['figures\MR_image\GNIMC',filename,'.png']); 
close(gcf);


%% 新增：表格输出 
fprintf('\n---------------------- Algorithm Comparison ----------------------\n');
fprintf('|     Algorithm      | Relative Error |     PSNR      | Execution Time (s) |\n');
fprintf('|----------------------|----------------|--------------------|\n');
fprintf('| AMM                  | %.3f         | %.3f         | %.3f               |\n', err_AMM, psnr_AMM, time_AMM);
fprintf('| Fnorm                | %.3f         | %.3f         | %.3f               |\n', err_Fnorm, psnr_Fnorm, time_Fnorm);
fprintf('| IMC (Maxide)         | %.3f         | %.3f         | %.3f               |\n', err_Maxide, psnr_Maxide, time_Maxide);
% fprintf('| DirtyIMC             | %.3f         | %.3f         | %.3f               |\n', err_DirtyIMC, time_DirtyIMC);
fprintf('| SGIMC                | %.3f         | %.3f         | %.3f               |\n', err_SGIMC, psnr_SGIMC, time_SGIMC);
fprintf('| MC (alm_MC)          | %.3f         | %.3f         | %.3f               |\n', err_MC, psnr_MC, time_MC);
fprintf('| GNIMC                | %.3f         | %.3f         | %.3f               |\n', err_GNIMC, psnr_GNIMC, time_GNIMC);
fprintf('-------------------------------------------------------------------\n');

%% 创建一个表格
algorithms = string({'AMM', 'Fnorm', 'IMC (Maxide)', 'SGIMC', 'MC (alm_MC)', 'GNIMC'}).';
error_vars = [err_AMM, err_Fnorm, err_Maxide, err_SGIMC, err_MC, err_GNIMC];
time_vars = [time_AMM, time_Fnorm, time_Maxide, time_SGIMC, time_MC, time_GNIMC];
psnrs = [psnr_AMM, psnr_Fnorm, psnr_Maxide, psnr_SGIMC, psnr_MC, psnr_GNIMC];
data = table(algorithms, num2cell(error_vars(:)), num2cell(psnrs(:)), num2cell(time_vars(:)));
data.Properties.VariableNames = {'Algorithm', 'Relative Error', 'PSNR', 'Execution Time (s)'};


% 将表格写入 Excel 文件
exlname = ['images' filename, num2str(SR), '_',num2str(delta), '.xlsx'];
writetable(data, exlname);
% [EOF]