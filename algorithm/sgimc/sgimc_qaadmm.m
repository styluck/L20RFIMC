% SGIMC MATLAB port – faithful QA-ADMM
% -------------------------------------------------------------
% This file is a **line-by-line MATLAB translation** of the original
% `sgimc_by_qaadmm_prototype.ipynb` experiment from the SGIMC repository
% (https://github.com/premolab/SGIMC).  The mathematical objective and
% the Quadratic-Approximation ADMM (QA-ADMM) update scheme are preserved
% exactly; we only swap Python/Numpy calls for their MATLAB equivalents.
% No simplifications to the algorithm are made – regularisation, dual
% variables, stopping tests, LBFGS inner solver, and the shrinkage
% operator all follow the original source.
%
% Usage (regression / squared-loss):
%   [W,H,history] = sgimc_qaadmm(X, Y, R, k, params);
%
% Inputs
%   X  : (n1×d1) row-side features      (double, sparse/dense)
%   Y  : (n2×d2) column-side features   (double, sparse/dense)
%   R  : (n1×n2) sparse rating matrix   (observed entries only)
%   k  : latent rank (integer > 0)
%   params : struct with hyper-parameters (mirrors python defaults):
%       .lambda_l1      – lasso weight                (1.0)
%       .lambda_group   – row-group l2 weight         (0.1)
%       .lambda_ridge   – ridge (ℓ2) weight           (1.0)
%       .eta            – ADMM penalty parameter      (1.0)
%       .maxOuterIter   – # Gauss–Seidel sweeps        (50)
%       .maxInnerIter   – # ADMM steps per sweep       (50)
%       .rtol, .atol    – ADMM stopping tolerances    (1e-5,1e-8)
%       .loss           – 'squared' (default) | 'logistic'
%       .verbose        – true/false (default true)
%
% Outputs
%   W, H   : learned coefficient matrices (d1×k, d2×k)
%   history: struct with objective values, residuals, compute time
% ---------------------------------------------------------------------
% Author: Lianghai Xiao 
% ---------------------------------------------------------------------

function [W,H,history] = sgimc_qaadmm(X, Y, R, k, params, lamb)
params.lambda_l1 = 0;
params.lambda_group = lamb;
params.lambda_ridge = 0;
% ----------------------- parameter defaults ---------------------------
if nargin < 5;                params = struct();            end
% === Set SGIMC parameters with defaults =============================
if ~isfield(params, 'lambda_l1') || isempty(params.lambda_l1)
    C_lasso = 1.0;
else
    C_lasso = params.lambda_l1;
end

if ~isfield(params, 'lambda_group') || isempty(params.lambda_group)
    C_group = 0.1;
else
    C_group = params.lambda_group;
end

if ~isfield(params, 'lambda_ridge') || isempty(params.lambda_ridge)
    C_ridge = 1.0;
else
    C_ridge = params.lambda_ridge;
end

if ~isfield(params, 'eta') || isempty(params.eta)
    eta = 1.0;
else
    eta = params.eta;
end

if ~isfield(params, 'maxOuterIter') || isempty(params.maxOuterIter)
    maxOuter = 10;
else
    maxOuter = params.maxOuterIter;
end

if ~isfield(params, 'maxInnerIter') || isempty(params.maxInnerIter)
    maxInner = 10;
else
    maxInner = params.maxInnerIter;
end

if ~isfield(params, 'rtol') || isempty(params.rtol)
    rtol = 1e-5;
else
    rtol = params.rtol;
end

if ~isfield(params, 'atol') || isempty(params.atol)
    atol = 1e-8;
else
    atol = params.atol;
end

if ~isfield(params, 'loss') || isempty(params.loss)
    lossType = 'squared';
else
    lossType = params.loss;
end

if ~isfield(params, 'verbose') || isempty(params.verbose)
    verbose = true;
else
    verbose = params.verbose;
end


[n1,d1] = size(X);   [n2,d2] = size(Y);
if issparse(R); Omega = spones(R); else; Omega = double(R~=0); end

% ----------------------------- init -----------------------------------
W = randn(d1,k)*0.01;   H = randn(d2,k)*0.01;
objHist = zeros(maxOuter,1);  residHist = zeros(maxOuter,2);
X_old = W*H';
tStart = tic;
for sweep = 1:maxOuter
    % ---- (A) update W with H fixed -----------------------------------
    ObjW = @(Wmat) imc_loss_grad(Wmat,H);
    W = admm_step(W, ObjW, [C_lasso,C_group,C_ridge], eta, maxInner, rtol, atol);

    % ---- (B) update H with W fixed  (transposed problem) --------------
    ObjH = @(Hmat) imc_loss_grad(Hmat,W,'transpose');
    H = admm_step(H, ObjH, [C_lasso,C_group,C_ridge], eta, maxInner, rtol, atol);
    X_ = W*H';
    measure = norm(X_-X_old,'fro')/max(1,norm(X_,'fro'));
    % ---- Diagnostics --------------------------------------------------
    [objVal,~] = imc_loss_grad(W,H);   % full objective value
    objHist(sweep) = objVal;
    if verbose
        fprintf('Sweep %3d | Obj %.4e | Δ %.2e | t=%.1fs\n', sweep, objVal, ...
            sweep>1 && abs(objHist(sweep)-objHist(sweep-1))/max(1,objHist(sweep-1)), toc(tStart));
    end
    if (measure<1.0e-5) ||(sweep>=10&&max(abs(objHist(sweep)-objHist(sweep-9)))<=1.0e-6*max(1,objHist(sweep)))
%         abs(objHist(sweep)-objHist(sweep-1)) <= 1e-4*objHist(sweep-1)
        objHist = objHist(1:sweep);
        residHist = residHist(1:sweep,:);
        break;
    end
    X_old = X_;
end

history.obj = objHist;  history.resid = residHist;  history.time = toc(tStart);

% --------------------------- nested fns -------------------------------
    function [f, g] = imc_loss_grad(AA, BB, mode)
        % IMC predictive matrix M = (X*AA)*(Y*BB)^T   (or transposed form)
        if nargin < 3; mode = 'normal'; end
        switch mode
            case 'normal'    % grad w.r.t AA (i.e., W)
                Afeat = X*AA;  Bfeat = Y*BB;
                M = Afeat * Bfeat';
                Err = (M - R) .* Omega;    % observed error
                switch lossType
                    case 'squared'
                        f = 0.5 * sum(Err(:).^2) + 0.5*C_ridge*(sum(AA(:).^2)+sum(BB(:).^2));
                        g = X'*(Err*Bfeat) + C_ridge*AA;
                    case 'logistic'
                        P = 1./(1+exp(-M));
                        f = -sum( R(:).*log(P(:)+eps) + (1-R(:)).*log(1-P(:)+eps) ) + ...
                            0.5*C_ridge*(sum(AA(:).^2)+sum(BB(:).^2));
                        g = X'*(((P-R).*Omega)*Bfeat) + C_ridge*AA;
                end
            case 'transpose' % grad w.r.t AA := H, with rows/cols swapped
                Afeat = Y*AA;  Bfeat = X*BB;   % note swap X<->Y
                M = Bfeat * Afeat';            % equals (XW)(YH)^T
                Err = (M - R) .* Omega;
                switch lossType
                    case 'squared'
                        f = 0.5*sum(Err(:).^2) + 0.5*C_ridge*(sum(AA(:).^2)+sum(BB(:).^2));
                        g = Y'*(Err'*Bfeat) + C_ridge*AA;
                    case 'logistic'
                        P = 1./(1+exp(-M));
                        f = -sum( R(:).*log(P(:)+eps) + (1-R(:)).*log(1-P(:)+eps) ) + ...
                            0.5*C_ridge*(sum(AA(:).^2)+sum(BB(:).^2));
                        g = Y'*(((P-R).*Omega)'*Bfeat) + C_ridge*AA;
                end
        end
    end

    function Z = admm_step(Z0, ObjFun, Cvec, eta_par, nIter, rtol_par, atol_par)
        C_l1 = Cvec(1);  C_grp = Cvec(2);  C_rdg = Cvec(3);
        Z  = Z0;          ZZ = Z0;          LL = zeros(size(Z0));
        nSq = sqrt(numel(Z0));
        for ii = 1:nIter
            Z_old  = Z;   ZZ_old = ZZ;
            % prox - regulariser (shrink)
            ZZ = shrink(Z + LL, C_l1*eta_par, C_grp*eta_par);
            % prox - loss + ridge  (LBFGS via fminunc)
            D  = ZZ - LL;   % centre for quadratic term
            subObj = @(vecW) fused_obj(vecW, D, ObjFun, C_rdg, eta_par);
            opts = optimoptions('fminunc','Algorithm','quasi-newton','GradObj','on', ...
                               'Display','off','MaxIter',20,'TolFun',1e-8,'TolX',1e-8);
            Zvec = fminunc(subObj, Z_old(:), opts);
            Z = reshape(Zvec, size(Z_old));
            % dual update
            LL = LL + (Z - ZZ);
            % residuals
            res_p = norm(Z(:)-ZZ(:));
            res_d = norm(ZZ(:)-ZZ_old(:)) / eta_par;
            tol_p = nSq*atol_par + rtol_par*max(norm(Z(:)), norm(ZZ(:)));
            tol_d = nSq*atol_par + rtol_par*norm(LL(:));
            if res_p <= tol_p && res_d <= tol_d
                break; end
        end
    end

    function [val, grad] = fused_obj(wvec, Dmat, ObjFun, C_rdg_par, eta_par)
        Wmat = reshape(wvec, size(Dmat));
        [f_loss, g_loss] = ObjFun(Wmat);
        diff = Wmat - Dmat;
        val = f_loss + 0.5*C_rdg_par*sum(Wmat(:).^2) + 0.5*eta_par*sum(diff(:).^2);
        grad = g_loss + C_rdg_par*Wmat + eta_par*diff;
        grad = grad(:);
    end

    function Y = shrink(Xin, l1, l2)
        % element-wise soft threshold (lasso)
        Y = sign(Xin).*max(abs(Xin)-l1, 0);
        % row-wise group l2 shrink
        rowN = sqrt(sum(Y.^2,2));
        scale = max(0, 1 - l2./(rowN+eps));
        Y = Y .* scale;
    end
end
