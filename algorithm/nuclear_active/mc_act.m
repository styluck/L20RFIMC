function [U, V, obj_vals, test_rmses, timelist]=mc_act(GXobs,lambda,GXtest, maxiter, maxk)
% [U, V, obj_vals, test_rmses, timelist] = mc_activealt(GXobs, lambda, GXtest, maxk, maxiter, mode)
%
% The active_alt algorithm for matrix completion problem; 
% solve the following optimization problem: 
%     
%     min_{X} 1/2 \sum_{i,j\in Omega} (A_{i,j} - X_{i,j})^2 + lambda * \|X\|_*
%
% Note: we recommend to shift the matrix to have zero mean before doing matrix completion 
% (since we use zero as initial solution). 
%
% Input Arguments: 
%   GXobs: observed rating matrix (A) -- a sparse matrix. 
%   lambda: regularization parameter -- a positive number. 
%   GXtest: testing ratings -- a sparse matrix. 
%   maxiter: maximum number of iterations -- a positive number. 
%   maxk: maximum number of rank -- a real number. 
%
% Output ArgumentS: 
%   U, V: The optimal X = UV'  
%   obj_vals: the history of objective function value for each iteration. 
%   test_rmses: the list of testing rmse for each iteration. 
%   timelist: the list of acummulated run time for each iteration. 


maxNumCompThreads(1);
  
[m n] = size(GXobs);
%obj_vals=zeros(maxiter,1);
GXobs=sparse(GXobs);
[i_row, j_col, data]=find(GXobs); number_observed=length(data);
[i_row_test, j_col_test, data_test]=find(GXtest); number_observed_test=length(data_test);

%% initial from zero
U=zeros(m,1); V=zeros(n,1);
S=zeros(1,1);
R = GXobs;
objval_old = norm(R,'fro')^2/2;
initial_k = 10;

i=0; 
timesvd = 0;
totaltime = 0;
oldV = [];
nextk = initial_k;
while (i<maxiter) 
	timebegin = cputime;
	i=i+1 ;
	
	kk = min(maxk, nextk);
	[u,s,v] = randomsvd(R, U, V, m, n, kk, oldV, 3);
	oldV = v;
	sing_vals=diag(s); 
	clear s;

	tmp=max(sing_vals-lambda,0);
	soft_singvals=tmp(tmp>0);
	no_singvals=length(soft_singvals);

	if (no_singvals == kk)
		nextk = ceil(kk*1.4);
	else
		nextk = no_singvals;
	end

	S = diag(soft_singvals);
	U = u(:,tmp>0); clear u;
	V = v(:,tmp>0); clear v;


	kk =  size(U,2);
	Z = S;
	for inner = 1:2
		%% update S
		R = CompResidual(GXobs, (U*S)', V');
		A = inv(Z);
		AA = A+A';
		grad =  U'*R*V - 0.5*lambda*(AA)*S;
		xx = zeros(kk,kk);
		r = grad;
		p = r;
		k = 0;
		rnorm = norm(r,'fro')^2;
		init_rnorm = rnorm;
		for cgiter = 1:4
			DUT = (V*p')';
			VDUT_Omega = CompProduct(GXobs, U', DUT);
			Ap = U'*VDUT_Omega*V+0.5*lambda*(p*AA);
			alpha = norm(r,'fro')^2/(sum(sum(p.*Ap)));
			xx = xx + alpha*p;
			r = r-alpha*Ap;
			rnorm_new = norm(r,'fro')^2;
%			fprintf('Inner iter %g, residual norm: %g\n', cgiter, rnorm_new);
			if ( rnorm < init_rnorm*1e-4)
				break;
			end
			beta = rnorm_new/rnorm;
			rnorm = rnorm_new;
			p = r+beta*p;
		end
		S = S + xx;

		%% Update Z
		Z = (S*S')^(0.5);
	end

	[u s v] = svd(S);
	U=U*u(:,1:kk);
	V=V*v(:,1:kk);
	S = s(1:kk,1:kk);

	totaltime = totaltime + cputime - timebegin;
	timelist(i) = totaltime;

	U = U*S;
	[uuuu ssss vvvv] = svd(S);

	R = CompResidual(GXobs, U', V');
	train_err = norm(R,'fro')^2;

	objval_new = train_err/2 + lambda*sum(diag(ssss));
	obj_vals(i)=objval_new;
	objval_old=objval_new;

	tttemp_test=project_obs_UV(U,V,i_row_test,j_col_test,number_observed_test);
	tttemp_test =  tttemp_test-data_test;
	test_err = sqrt(tttemp_test'*tttemp_test/numel(data_test));
	test_rmses(i) = test_err;
	%fprintf('Iter %g time %g obj %g test_rmse %g\n', i, totaltime, objval_new, test_err);
end

