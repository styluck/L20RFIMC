%% ********************************************************************
%  filename: SInf_AMM_CutCol
%% **********************************************************************
%% Majorized alternating proximal minimization method for solving
%  
%  min{0.5||P_Omega(M-AUV'B')||^2 + 0.5*mu(||U||_F^2+||V||_F^2)}+ 0.5*lambda(||U||_{2,0}+||V||_{2,0}) (*)
%  
%  in the order: U--->V---->U
%% **************************************************************************
%%  2025-01-24  by TingTao
%% *************************************************************************

function [rankV,AXB] = SInf_AMM_CutCol(Mstar,bb,P,Q,d,nzidx,A,B,lambda,pars,OPTIONS)

if isfield(OPTIONS,'printyes');   printyes  = OPTIONS.printyes;    end
if isfield(OPTIONS,'maxiter');    maxiter   = OPTIONS.maxiter;     end
if isfield(OPTIONS,'tol');        tol   =  OPTIONS.tol;     end
if isfield(OPTIONS,'Lip_const');  eta   =  OPTIONS.Lip_const;     end

mu = 1.0e-8;        gamma = 1.0e-8;

lgamma = 1.0e-8;    gratio = 0.8;

gamma1 = gamma;     gamma2 = gamma;

if  (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************** MAPM for low-rank recovery problems ***********************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter      rankX       relerr      measure         obj      time ');
end

%% ***************** Initialization **************************

nr = pars.nr;     nc = pars.nc;   r = pars.r; 

dsqrt = d.^(1/2);

gam1_mu = gamma1+mu;     gam2_mu = gamma2+mu;

AXBnz = zeros(nr,nc);    AXB = (A*P.*d)*(B*Q)';   % d=eyes(r)

AXBnz(nzidx) = AXB(nzidx) - bb; 

temp_Gk = P.*(eta*d) - A'*AXBnz*(B*Q) + gamma1*P;  %temp_Gk = (eta*P.*d*Q'- A'*AXBnz*B)*Q + gamma1*P;

%% ************************* Main Loop *********************************

rank_list = zeros(maxiter,1);

tstart = clock;

rankV = r;

for iter=1:maxiter
    
    %% ********************* Given B, to solve A **********************
  
    dmu_sqrt = (eta*d+gam1_mu).^(1/2);
    
    bk = dsqrt./dmu_sqrt;
    
    temp_Gk_Cnorm = sum(temp_Gk.*temp_Gk);
    
    Gk_cnorm = temp_Gk_Cnorm.*bk;
    
    ind = Gk_cnorm>lambda;
 
    rankU = sum(ind);
    
    if (rankU==0)
        
        disp('lambda is too large,please reset it again')
        
        Xopt =0; rankV = 0;
        
        return;
    end
    
    UD = temp_Gk(:,ind).*(bk(ind)./dmu_sqrt(ind).*dsqrt(ind));
    
    [H,D,L] = svd(UD,'econ');
    
    P = H;           Q = Q(:,ind)*L;
    
    d = diag(D)';    dsqrt = d.^(1/2);

    Pd = P.*d;
    
    AXB = (A*Pd)*(B*Q)';
    
    AXBnz(nzidx) = AXB(nzidx)-bb;

    Xold = Pd*Q';  %  X = Pd*Q'; 

    %% ****************** Given A, to solve B *************************

    temp_Hk = Q.*(eta*d) - ((A*P)'*AXBnz*B)' + gamma2*Q;   % temp_Hk  = (eta*X - A'*AXBnz*B)'*P + gamma2*Q;
    
    dmu_sqrt = (eta*d+gam2_mu).^(1/2);
    
    bk = dsqrt./dmu_sqrt;
    
    temp_Hk_cnorm = sum(temp_Hk.*temp_Hk);
    
    Hk_cnorm = temp_Hk_cnorm.*bk;
    
    ind = Hk_cnorm>lambda;
    
    rankV = sum(ind);

    VD = temp_Hk(:,ind).*(bk(ind)./dmu_sqrt(ind).*dsqrt(ind));
    
    [H,D,L] = svd(VD,'econ');
    
    P = P(:,ind)*L;   Q = H;
    
    d = diag(D)';     dsqrt = d.^(1/2);

    Pd = P.*d;
    
    X = Pd*Q';    AXB = (A*Pd)*(B*Q)';   % X=UV', 
    
    AXBnz(nzidx) = AXB(nzidx) - bb;  

    temp_Gk = eta*Pd - A'*(AXBnz*(B*Q)) + gamma1*P;  % temp_Gk =  (eta*X-A'*AXBnz*B)*Q + gamma1*P;
    
    rank_list(iter) = rankV;
     
    obj = 0.5*norm(AXBnz(nzidx))^2 + 0.5*mu*sum(d) + lambda*rankV;
    
    obj_list(iter) = obj;
    
    %% **************** check the stopping criterion ******************
    
    time = etime(clock,tstart);
    
    measure = norm(X-Xold,'fro')/max(1,norm(X,'fro'));
    
    if (printyes)&&(mod(iter,10)==0||iter<=10)
             
        relerr = norm(AXB-Mstar,'fro')/norm(Mstar,'fro');
        
        fprintf(' \n %2d          %2d        %3.2e      %3.2e      %3.5e     %3.2f \n',iter,rankV,relerr,measure,obj,time);
        
    end
  
    if (measure<tol) ||(iter>=10&&max(abs(obj-obj_list(iter-9:iter)))<=1.0e-6*max(1,obj)) %&& max(abs(rankV-rank_list(iter-19:iter)))<=1.0e-8)             
        return;
    end
    
    gamma1 = max(lgamma,gratio*gamma1);
    
    gam1_mu = gamma1 + mu;
    
    gamma2 = max(lgamma,gratio*gamma2);
    
    gam2_mu = gamma2 + mu;
    
end

if (iter==maxiter)
    
    fprintf('\n maxiter');
    
end

end

