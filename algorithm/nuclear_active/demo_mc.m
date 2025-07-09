randn('state',2009);
rand('state',2009);
maxNumCompThreads(1)
load movielens100kbrenorm
lambda = 15;
k = 100;


[U1, V1, obj_vals_alt, test_rmse_alt, timelist_alt]=mc_alt(Y',lambda, Yt', 30, k);


