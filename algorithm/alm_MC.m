function [X] = alm_MC(M, Omega, mu, maxIter,isDisp)

tol = 1e-3;tolProj = 1e-5;
[m n] = size(M);

Omega_c = ones(m,n) - Omega;

I = eye(n);


% initialize

X = zeros( m, n);
E = zeros( m, n);
Y = zeros( m, n);
R = zeros( m, n);



total_svd = 0;


iter = 0;
iter2 = 0;
converged = false;
sv = 5;
svp = sv;
max_sv = min(m,n);
while ~converged       
    iter = iter + 1;
    
    % solve the primal problem by alternative projection
    primal_converged = false;
    primal_iter = 0;
    sv = min(sv + round(max_sv * 0.1),max_sv);
	
    while primal_converged == false && primal_iter <10
              
        A=M+E-(1/mu)*Y;    
		
        if choosvd(n, sv) == 1
            [U Sig V] = lansvd(A, sv, 'L');
        else
            [U Sig V] = svd(A, 'econ');
        end
        diagSig = diag(Sig);
        svp = length(find(diagSig > 1/mu));
		
        if svp < sv
            sv = min(svp + 1, max_sv);
        else
            sv = min(svp + round(0.05*max_sv), max_sv);
        end
		
        temp_X = U(:,1:svp)*diag(diagSig(1:svp)-1/mu)*V(:,1:svp)';    
        
        E = Omega_c.*(temp_X-M+(1/mu)*Y);
                       
        if norm(X - temp_X, 'fro') < tolProj*norm(X,'fro');
            primal_converged = true;
        end
		
        X = temp_X;
        primal_iter = primal_iter + 1;
        total_svd = total_svd + 1;
               
    end
     
    temp_R = X - M  - E;      
    Y = Y + mu*temp_R;
    
    %% stop Criterion    
    stopCriterion = norm(temp_R - R, 'fro');
    if stopCriterion < tol*norm(R,'fro')
        converged = true;
    end    
    R = temp_R;
   
   if isDisp
       disp(['Iteration' num2str(iter) ' #svd ' num2str(total_svd) ' Rank(X) ' num2str(svp)...
            ' stopCriterion ' num2str(stopCriterion)  ]);
   end
    
    if ~converged && iter >= maxIter
        %disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end


