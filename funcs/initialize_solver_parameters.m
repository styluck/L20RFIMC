function [pars,U1,V1, U2,V2] = initialize_solver_parameters(nr, nc, d1, d2, r, Mstar, A, B, nzidx, bb, lamb)
% Initialize solver parameters
pars.lambda = lamb;
pars.mu = 1e1;
pars.eta = 0.09 * (norm(randn(nr, 2 * r), 2) * norm(randn(nc, 2 * r), 2))^2;
pars.gamma = 1.0e-8;

pars.n1 = nr;
pars.n2 = nc;

pars.d1 = d1;
pars.d2 = d2;

pars.normM = norm(Mstar, 'fro');
pars.normb = norm(bb);

pars.maxiter = 2000;
pars.verbose = 1;
pars.tol = 1.0e-6;
pars.type = 0;

% generate initial points
% U1 = randn(d1, r);
% V1 = randn(d2, r);
% U1 = orth(U1);    
% V1 = orth(U2);

M = zeros(nr, nc);

M(nzidx) = bb;
AMB = A'*M*B;
[U1,dM,V1] = lansvd(full(AMB),r,'L',1.0e-6);

dM = diag(dM)';
U1 = U1(:,1:r);
V1 = V1(:,1:r);

U2 = U1.*dM(1:r).^(1/2);      
V2 = V1.*dM(1:r).^(1/2);


end

% [EOF]