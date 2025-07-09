%% ********************************************************************
%  filename: AMM_Fnorm
%
%% **********************************************************************
%% Alternating majorization-minimization method for solving
%  
%  min{0.5||P_Omega(M-AUV'B')||^2 + 0.5*lambda(||U||_F^2+||V||_F^2)} (*)
%  
%  in the order: U--->V---->U
%
%% **********************************************************************
%%  2025-04-2 by TaoTing
%% *************************************************************************

function [AXB,rankX,Loss]= AMM_Fnorm(Mstar,b,U,V,nzidx,A,B,OPTIONS,pars,lambda)

if isfield(OPTIONS,'tol');        tol       = OPTIONS.tol;         end
if isfield(OPTIONS,'printyes');   printyes  = OPTIONS.printyes;    end
if isfield(OPTIONS,'maxiter');    maxiter   = OPTIONS.maxiter;     end
if isfield(OPTIONS,'LipX');   LipX      = OPTIONS.LipX;    end

nr= pars.nr;    nc = pars.nc;  

LipU = LipX*norm(V,2)^2;      tauU = LipU + lambda ; 

if  (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************** AMM_solver  for  solving low-rank recovery  problems  ********************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter     optmeasure     rankX      fval       time ');
end

%% ************************* Main Loop *********************************

tstart = clock;

obj_list = zeros(maxiter,1);

rank_list = zeros(maxiter,1);

PX = zeros(nr,nc);

X = U*V';

[~,gradU] = funU(U,V,b,A,B,nzidx,PX);

for iter = 1:maxiter
    
  %% ****************** to compute Unew *************************

    Unew = (1/tauU)*(LipU*U-gradU);

    LipV = LipX*norm(Unew,2)^2;     tauV = LipV + lambda;
             
   %% ****************** to compute Vnew **************************
    
    [~,gradV] = funV(Unew,V,b,A,B,nzidx,PX);
    
    Vnew = (1/tauV)*(LipV*V-gradV);  

    d  = svd(Vnew); 
    
    rankX = sum(d>max(d)*1.0e-6);

    rank_list(iter) = rankX;
        
%% ******************** Optimality checking ************************
    
    [Xnew,AXB,Loss,gradU] = funUV(Unew,Vnew,b,A,B,nzidx,PX);
    
    obj = Loss + 0.5*lambda*(norm(Unew,'fro')^2 + norm(Vnew,'fro')^2) ;
           
    obj_list(iter) = obj;
    
    ttime = etime(clock,tstart);

    measure = norm(X-Xnew,'fro')/max(1,norm(X,'fro'));
    
    if (printyes&&mod(iter,10)==1 )

        fprintf('\n  %2d      %3.2e     %2d        %3.4e     %3.2f',iter,measure,rankX,obj,ttime);
         
%         relerr = norm(AXB-Mstar,'fro')/norm(Mstar,'fro');
        
    end

    if (iter>10)&&( (measure<tol) ||(max(abs(obj-obj_list(iter-9:iter)))<=1.0e-6*max(1,obj)))

        return;
    end

   U = Unew;    V = Vnew;   X = Xnew;

   LipU = LipX*norm(Vnew,2)^2;     tauU = LipU + lambda;
    
end

if (iter==maxiter)
   
      fprintf('\n maxiter'); 
end

end

