
!leave this line blank
memory, 248.75, m
symmetry, nosym
basis={

!
! He        (4s) -> [2s]
! He        (4s) -> [2s]
s,  HE ,  38.421634  ,5.77803  ,1.241774  ,0.297964
c,  1.3,  0.023766,  0.154679,  0.46963 
c,  4.4,  1.0 }
geometry={He  0.0  0.0  0.0}
dummy,
charge=0
spin=0

{matrop                   !read the modified core hamiltonian
read,h01,type=h0,file=3__h0.mat
save,h01,7500.1,h0}

{hf;noenest}
