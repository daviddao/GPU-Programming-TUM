function [P1,Pt1, PX, L]=cpd_MATLAB(X,T, sigma2 ,outlier);

ksig = -2.0 * sigma2;
outlier_tmp=(outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-outlier)*N); 



end