
!leave this line blank
memory, 248.75, m
symmetry, nosym
basis={

!
! Li        (10s,4p) -> [3s,2p]
! Li        (10s,4p) -> [3s,2p]
s,  LI ,  642.41892  ,96.798515  ,22.091121  ,6.2010703  ,1.9351177  ,0.6367358  ,2.3249184  ,0.6324306  ,0.0790534  ,0.035962
c,  1.6,  0.0021426,  0.0162089,  0.0773156,  0.245786,  0.470189,  0.3454708 
c,  7.9,  -0.0350917,  -0.1912328,  1.0839878 
c,  10.10,  1.0 
p,  LI ,  2.3249184  ,0.6324306  ,0.0790534  ,0.035962
c,  1.3,  0.0089415,  0.1410095,  0.9453637 
c,  4.4,  1.0 }
geometry={Li  0.0  0.0  0.0}
dummy,
charge=1
spin=0

{matrop                   !read the modified core hamiltonian
read,h01,type=h0,file=1__h0.mat
save,h01,7500.1,h0}

{rhf;noenest}
{fci;core;dump}

