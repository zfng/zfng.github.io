# import packages ---------------------------------------------------------------------------------------------------------------------
import math
import scipy.special as sp
import numpy as np
import random


# declare data values -------------------------------------------------------------------------------------------------------------------
N = 891 + 418 # = 1309
N_S = 500
N_F = 314 + 152 # = 466
N_M = 577 + 266 # = 843

# declare data values for females -------------------------------------------------------------------------------------------------------

N_FA = 277 + 142 # = 419
N_FC = 37 + 10 # = 47

NO_F = 233
NO_FA = 277
NO_SFA = 213
NO_SFC = 20

N_FAYC = 161 + 77 # = 238
N_FCYC = 9 + 3 # = 12
N_FANC = 116 + 65 # = 181
N_FCNC = 28 + 7 # = 35

NO_FAYC = 161
NO_FCYC = 9
NO_FANC = 116
NO_FCNC = 28

NO_SFAYC = 153
NO_SFCYC = 8
NO_SFANC = 60
NO_SFCNC = 12


# FAYC formula ------------------------------------------------------------------------------------------------------------------------

def prob_FAYC(N, N_S, N_M, N_F, N_FA, N_FAYC, N_FANC, NO_SFA, NO_SFAYC, NO_FAYC):
    
    prob_FAYC = 0           
    
    for N_SF in range(math.floor(NO_SFA/(N_FA/N_F)), N_F):
        
        # A brief explanation on how the ranges of the sums are determined
        
        # Given N_SFA, we assign N_SFAYC a uniform distribution on (N_SFA*(N_FAYC/N_FA), N_S). Given the data, we also have
        # NO_SFAYC <= N_SFAYC <= N_FAYC, so max(NO_SFAYC, N_SFA*(N_FAYC/N_FA)) <= N_SFAYC <= min(N_S, N_FAYC).
        # Note N_FAYC =238, N_S = 500, and the bounds on N_SFAYC always make sense. 
        # Suppose NO_SFAYC <= N_SFA*(N_FAYC/N_FA), then we need N_SFA*(N_FAYC/N_FA) <= N_FAYC, which is clearly true
        # Suppose N_SFA*(N_FAYC/N_FA) <= NO_SFAYC, then we need NO_SFAYC <= N_FAYC, which is also true (NO_SFAYC <= N_SFAYC <= N_FAYC)
        # Next, given N_SF, we assign N_SFA uniform on (0, N_SF*(N_FA/N_F)). Also, given the data, NO_SFA <= N_SFA, so 
        # NO_SFA <= N_SFA <= N_SF*(N_FA//N_F)
        # Finally, we assign N_SF uniform on (N_S*(N_F/N), N_S). With the data, and that the bounds on N_SFA should make sense
        # then max(NO_SF, N_S*(N_F/N), NO_SFA/(N_FA/N_F)) <= N_SF <= min(N_S, N_F)
        # As before, NO_SF = 233, N_S*(N_F/N) = 178, NO_SFA/(N_FA/N_F) = 237, so NO_SFA/(N_FA/N_F) <= N_SF <= N_F
        
        for N_SFA in range(NO_SFA, math.floor(N_SF*(N_FA/N_F))):
            
            
            for N_SFAYC in range(math.floor(max(NO_SFAYC, N_SFA*(N_FAYC/N_FA))), N_FAYC):
        
                # compute the survival probability of a cabin class adult female in the test set given
                # the number of such survivors
        
                probN_FAYC = (N_SFAYC - NO_SFAYC) / (N_FAYC - NO_FAYC)
        
                # compute the survival probability of NO_FAYC cainb class adult females in the training
                # set given the number of such survivors
        
                probD_FAYC = np.nan_to_num(sp.binom(N_SFAYC, NO_SFAYC)) * \
                            np.nan_to_num(sp.binom(N_FAYC-N_SFAYC, NO_FAYC-NO_SFAYC)) / np.nan_to_num(sp.binom(N_FAYC, NO_FAYC))
    
                # compute the product of the densities of the prior distribution for N_SF, N_SFA, N_SFAYC
            
                probP_FAYC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFA/N_FA)*N_FANC))
                
                Tprob_FAYC = probN_FAYC * probD_FAYC * probP_FAYC
        
                prob_FAYC = prob_FAYC + Tprob_FAYC 
        
    return prob_FAYC

# compute the probability of the training data given the prior information p(D_0|I); it's done by expanding in the values of N_SFAYC

def probTR_FAYC(N, N_S, N_M, N_F, N_FA, N_FAYC, N_FANC, NO_SFA, NO_SFAYC, NO_FAYC):
    
    probTR_FAYC = 0           
    
    for N_SF in range(math.floor(NO_SFA/(N_FA/N_F)), N_F):
     
        for N_SFA in range(NO_SFA, math.floor(N_SF*(N_FA/N_F))):
            
            
            for N_SFAYC in range(math.floor(max(NO_SFAYC, N_SFA*(N_FAYC/N_FA))), N_FAYC):
        
               
                probD_FAYC = np.nan_to_num(sp.binom(N_SFAYC, NO_SFAYC)) * \
                            np.nan_to_num(sp.binom(N_FAYC-N_SFAYC, NO_FAYC-NO_SFAYC)) / np.nan_to_num(sp.binom(N_FAYC, NO_FAYC))
    
                probP_FAYC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFA/N_FA)*N_FANC))
                
                TprobTR_FAYC = probD_FAYC * probP_FAYC
        
                probTR_FAYC = probTR_FAYC + TprobTR_FAYC 
        
    return probTR_FAYC

# the posterior probability for FAYC is taking the ratio of prob_FAYC(N, N_S, N_M, N_F, N_FA, N_FAYC, N_FANC, NO_SFA, NO_SFAYC, NO_FAYC)
# to probTR_FAYC(N, N_S, N_M, N_F, N_FA, N_FAYC, N_FANC, NO_SFA, NO_SFAYC, NO_FAYC); and it is the same for the other probabilities


# FANC formula -------------------------------------------------------------------------------------------------------------------------

def prob_FANC(N, N_S, N_M, N_F, N_FA, N_FANC, NO_SFA, NO_SFANC, NO_FANC):
    
    prob_FANC = 0           
    
    for N_SF in range(math.floor(NO_SFA/(N_FA/N_F)), N_F):
        
       
        # Given N_SFA, from the prior information that N_SFANC/N_FANC < N_SFAYC/N_FAYC, we assign N_SFANC a uniform 
        # distribution on (0, N_SFA*(N_FANC/N_FA)).
        # Also, given the data, NO_SFANC <= N_SFANC, so the bounds on N_SFANC are NO_SFANC <= N_SFANC <= N_SFA*(N_FANC/N_FA)
        # Next, given N_SF, from the prior information that N_SFA/N_FA < N_SFC/N_FC, we assign N_SFA a uniform distribution on
        # (0, N_SF*(N_FA/N_F)). Again, given the data, NO_SFA <= N_SFA, Moreover, we require the bounds on N_SFANC make sense,
        # so the bounds on N_SFA are max(NO_SFA, NO_SFANC/(N_FANC/N_FA)) <= N_SFA <= N_SF*(N_FA/N_F)
        # Substituting the given values to simplify the bounds, NO_SFA = 213, NO_SFANC/(N_FANC/N_FA) = 139, so
        # NO_SFA <= N_SFA <= N_SF*(N_FA/N_F)
        # Finally, by similar reasoning, we assign N_SF a uniform distribution on (N_S*(N_F/N), N_S)
        # That the bounds on N_SFA should make sense, we get NO_SFA/(N_FA/N_F) <= N_SF; also, NO_SF <= N_SF <= N_F
        # so the bounds are max(NO_SF, N_S*(N_F/N), NO_SFA/(N_FA/N_F)) <= N_SF <= min(N_S, N_F)
        # Substituting the given values to simplify the bounds, note NO_SF = 233, N_S*(N_F/N) = 178, NO_SFA/(N_FA/N_F) = 237
        # N_S = 500, N_F = 466
        # Thus NO_SFA/(N_FA/N_F) <= N_SF <= N_F
        
        
        for N_SFA in range(NO_SFA, math.floor(N_SF*(N_FA/N_F))):
            
            
            for N_SFANC in range(NO_SFANC, math.floor(N_SFA*(N_FANC/N_FA))):
                
        
                    # compute the survival probability of a non-cabin class adult female in the test set given
                    # the number of such survivors
        
                    probN_FANC = (N_SFANC - NO_SFANC) / (N_FANC - NO_FANC)
        
                    # compute the survival probability of NO_SFANC non-cabin class adult females in the training
                    # set given the number of such survivors
        
                    probD_FANC = np.nan_to_num(sp.binom(N_SFANC, NO_SFANC)) * \
                                np.nan_to_num(sp.binom(N_FANC-N_SFANC, NO_FANC-NO_SFANC)) / np.nan_to_num(sp.binom(N_FANC, NO_FANC))
    
                    # compute the product of the prior distribution densities for N_SF, N_SFA, N_SFANC
        
                    probP_FANC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFA/N_FA)*N_FANC))
                
                    Tprob_FANC = probN_FANC * probD_FANC * probP_FANC
        
                    prob_FANC = prob_FANC + Tprob_FANC 
            
    return prob_FANC


def probTR_FANC(N, N_S, N_M, N_F, N_FA, N_FANC, NO_SFA, NO_SFANC, NO_FANC):
    
    probTR_FANC = 0           
    
    for N_SF in range(math.floor(NO_SFA/(N_FA/N_F)), N_F):
       
        for N_SFA in range(NO_SFA, math.floor(N_SF*(N_FA/N_F))):
            
            
            for N_SFANC in range(NO_SFANC, math.floor(N_SFA*(N_FANC/N_FA))):
        
                    probD_FANC = np.nan_to_num(sp.binom(N_SFANC, NO_SFANC)) * \
                                np.nan_to_num(sp.binom(N_FANC-N_SFANC, NO_FANC-NO_SFANC)) / np.nan_to_num(sp.binom(N_FANC, NO_FANC))
   
        
                    probP_FANC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFA/N_FA)*N_FANC))
                
                    TprobTR_FANC = probD_FANC * probP_FANC
        
                    probTR_FANC = probTR_FANC + TprobTR_FANC 
            
    return probTR_FANC

# FCYC formula ------------------------------------------------------------------------------------------------------------------------

def prob_FCYC(N, N_S, N_M, N_F, N_FC, N_FCNC, N_FCYC, NO_SF, NO_SFC, NO_SFCYC, NO_FCYC):
    
    prob_FCYC = 0           
    
    for N_SF in range(NO_SF, N_F):
        
        # All bounds make sense. For example, NO_SFCYC <= N_FCYC, N_SFC*(N_FCYC/N_FC) <= N_FCYC. The others are similar
        
        for N_SFC in range(math.floor(max(NO_SFC, N_SF*(N_FC/N_F))), N_FC):
            
            
            for N_SFCYC in range(math.floor(max(NO_SFCYC, N_SFC*(N_FCYC/N_FC))), N_FCYC):
        
                probN_FCYC = (N_SFCYC - NO_SFCYC) / (N_FCYC - NO_FCYC)
        
                probD_FCYC = np.nan_to_num(sp.binom(N_SFCYC, NO_SFCYC)) * \
                            np.nan_to_num(sp.binom(N_FCYC-N_SFCYC, NO_FCYC-NO_SFCYC)) / np.nan_to_num(sp.binom(N_FCYC, NO_FCYC))
    
            
                probP_FCYC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFC/N_FC)*N_FCNC))
                
                Tprob_FCYC = probN_FCYC * probD_FCYC * probP_FCYC
        
                prob_FCYC = prob_FCYC + Tprob_FCYC 
            
    return prob_FCYC

def probTR_FCYC(N, N_S, N_M, N_F, N_FC, N_FCNC, N_FCYC, NO_SF, NO_SFC, NO_SFCYC, NO_FCYC):
    
    probTR_FCYC = 0           
    
    for N_SF in range(math.floor(max(NO_SF, N_S*(N_F/N))), N_F):
      
        for N_SFC in range(math.floor(max(NO_SFC, N_SF*(N_FC/N_F))), N_FC):
            
            
            for N_SFCYC in range(math.floor(max(NO_SFCYC, N_SFC*(N_FCYC/N_FC))), N_FCYC):
        
                
                probD_FCYC = np.nan_to_num(sp.binom(N_SFCYC, NO_SFCYC)) * \
                            np.nan_to_num(sp.binom(N_FCYC-N_SFCYC, NO_FCYC-NO_SFCYC)) / np.nan_to_num(sp.binom(N_FCYC, NO_FCYC))
    
            
                probP_FCYC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFC/N_FC)*N_FCNC))
                
                TprobTR_FCYC = probD_FCYC * probP_FCYC
        
                probTR_FCYC = probTR_FCYC + TprobTR_FCYC 
            
    return probTR_FCYC

# FCNC formula ------------------------------------------------------------------------------------------------------------------------

def prob_FCNC(N, N_S, N_M, N_F, N_FC, N_FCNC, NO_SF, NO_SFC, NO_SFCNC, NO_FCNC):
    
    prob_FCNC = 0           
    
    for N_SF in range(NO_SF, N_F):
        
        # Given N_SFC, we assign N_SFCNC a uniform distribution on (0, N_SFC*(N_FCNC/N_FC)). Since the summand is nonzero only if
        # NO_SFCNC <= N_SFCNC <= N_FCNC, the bounds on N_SFCNC are NO_SFCNC <= N_SFCNC <= N_SFC*(N_FCNC/N_FC)
        # Given N_SF, we assign N_SFC a uniform distribution on (N_SF*(N_FC/N_F), N_S). To sharpen the lower bound, we require
        # the bounds on N_SFCNC make sense that NO_SFCNC <= N_SFC*(N_FCNC/N_FC), so NO_SFCNC/(N_FCNC/N_FC) <= N_SFC. 
        # On the other hand, given the data, NO_SFC <= N_SFC, so max(NO_SFC, N_SF*(N_FC/N_F), NO_SFCNC/(N_FCNC/N_FC)) <= N_SFC <= min(N_S, N_FC)
        # Substituting the given values to simplify the bounds, NO_SFC = 20, NO_SFCNC/(N_FCNC/N_FC) = 17, so the bounds on N_SFC are
        # max(NO_SFC, N_SF*(N_FC/N_F)) <= N_SFC <= N_FC. Note the bounds always make sense, because NO_SFC <= N_FC and
        # N_SF*(N_FC/N_F) <= N_FC
        # Finally, we assign N_SF a uniform on (N_S*(N_F/N), N_S). Given the data, NO_SF <= N_SF, so 
        # max(NO_SF, N_S*(N_F/N)) <= N_SF <= min(N_S, N_F)
        # Substituing the given values, NO_SF = 233, and N_S*(N_F/N) = 178
        # So the bounds on N_SF are NO_SF <= N_SF <= N_F

        
        
        for N_SFC in range(math.floor(max(NO_SFC, N_SF*(N_FC/N_F))), N_FC):
            
            
            for N_SFCNC in range(NO_SFCNC, math.floor(N_SFC*(N_FCNC/N_FC))):
        
                probN_FCNC = (N_SFCNC - NO_SFCNC) / (N_FCNC - NO_FCNC)
        
               
                probD_FCNC = np.nan_to_num(sp.binom(N_SFCNC, NO_SFCNC)) * \
                            np.nan_to_num(sp.binom(N_FCNC-N_SFCNC, NO_FCNC-NO_SFCNC)) / np.nan_to_num(sp.binom(N_FCNC, NO_FCNC))
    
            
                probP_FCNC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFC/N_FC)*N_FCNC))
                
                Tprob_FCNC = probN_FCNC * probD_FCNC * probP_FCNC
        
                prob_FCNC = prob_FCNC + Tprob_FCNC 
            
    return prob_FCNC

def probTR_FCNC(N, N_S, N_M, N_F, N_FC, N_FCNC, NO_SF, NO_SFC, NO_SFCNC, NO_FCNC):
    
    probTR_FCNC = 0           
    
    for N_SF in range(NO_SF, N_F):
  
        for N_SFC in range(math.floor(max(NO_SFC, N_SF*(N_FC/N_F))), N_FC):
            
            
            for N_SFCNC in range(NO_SFCNC, math.floor(N_SFC*(N_FCNC/N_FC))):
        
              
                probD_FCNC = np.nan_to_num(sp.binom(N_SFCNC, NO_SFCNC)) * \
                            np.nan_to_num(sp.binom(N_FCNC-N_SFCNC, NO_FCNC-NO_SFCNC)) / np.nan_to_num(sp.binom(N_FCNC, NO_FCNC))
    
            
                probP_FCNC = (1/((N_S/N)*N_M)) * (1/((N_SF/N_F)*N_FA)) * (1/((N_SFC/N_FC)*N_FCNC))
                
                TprobTR_FCNC = probD_FCNC * probP_FCNC
        
                probTR_FCNC = probTR_FCNC + TprobTR_FCNC 
            
    return probTR_FCNC

# declare data values for males ---------------------------------------------------------------------------------------------------------

N_M = 577 + 266 # = 843 
N_MA = 537 + 249 # = 786
N_MC = 40 + 17 # = 57

NO_M = 577
NO_SM = 109
NO_SMA = 86
NO_SMC = 23

N_MAYC = 218 + 117 # = 335
N_MCYC = 12 + 3 # = 15
N_MANC = 319 + 132 # = 451
N_MCNC = 28 + 14 # = 42

NO_MAYC = 218
NO_MCYC = 12
NO_MANC = 319
NO_MCNC = 28

NO_SMAYC = 50
NO_SMCYC = 12
NO_SMANC = 36
NO_SMCNC = 11


# MAYC formula ------------------------------------------------------------------------------------------------------------------------

def prob_MAYC(N, N_S, N_M, N_MA, N_MANC, N_MAYC, NO_SM, NO_SMA, NO_SMAYC, NO_MAYC):
    
    prob_MAYC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
        
        # Substituting the given values to simplify the bounds, NO_SM = 109, NO_SMA/(N_MA/N_M) = 93
       
        for N_SMA in range(NO_SMA, math.floor(N_SM*(N_MA/N_M))):
            
            
            for N_SMAYC in range(math.floor(max(NO_SMAYC, N_SMA*(N_MAYC/N_MA))), N_MAYC):
        
              
                probN_MAYC = (N_SMAYC - NO_SMAYC) / (N_MAYC - NO_MAYC)
        
               
                probD_MAYC = np.nan_to_num(sp.binom(N_SMAYC, NO_SMAYC)) * \
                            np.nan_to_num(sp.binom(N_MAYC-N_SMAYC, NO_MAYC-NO_SMAYC)) / np.nan_to_num(sp.binom(N_MAYC, NO_MAYC))
    
            
                probP_MAYC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMA/N_MA)*N_MANC))
                
                Tprob_MAYC = probN_MAYC * probD_MAYC * probP_MAYC
        
                prob_MAYC = prob_MAYC + Tprob_MAYC 
            
    return prob_MAYC

def probTR_MAYC(N, N_S, N_M, N_MA, N_MANC, N_MAYC, NO_SM, NO_SMA, NO_SMAYC, NO_MAYC):
    
    probTR_MAYC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
       
        for N_SMA in range(NO_SMA, math.floor(N_SM*(N_MA/N_M))):
            
            
            for N_SMAYC in range(math.floor(max(NO_SMAYC, N_SMA*(N_MAYC/N_MA))), N_MAYC):
        
             
                probD_MAYC = np.nan_to_num(sp.binom(N_SMAYC, NO_SMAYC)) * \
                            np.nan_to_num(sp.binom(N_MAYC-N_SMAYC, NO_MAYC-NO_SMAYC)) / np.nan_to_num(sp.binom(N_MAYC, NO_MAYC))
    
            
                probP_MAYC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMA/N_MA)*N_MANC))
                
                TprobTR_MAYC = probD_MAYC * probP_MAYC
        
                probTR_MAYC = probTR_MAYC + TprobTR_MAYC 
            
    return probTR_MAYC

# MANC formula ------------------------------------------------------------------------------------------------------------------------

def prob_MANC(N, N_S, N_M, N_MA, N_MANC, NO_SM, NO_SMA, NO_SMANC, NO_MANC):
    
    prob_MANC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
        
        # Substituting the given values to simplify the bounds, NO_SMA = 86, NO_SMANC/(N_MANC/N_MA) = 63, the bounds on N_SMA are
        # NO_SMA <= N_SMA <= N_SM*(N_MA/N_M)
        # For N_SM, NO_SM = 109, NO_SMA/(N_MA/N_M) = 93, so NO_SM <= N_SM <= N_S*(N_M/N)
       
        for N_SMA in range(NO_SMA, math.floor(N_SM*(N_MA/N_M))):
            
            
            for N_SMANC in range(NO_SMANC, math.floor(N_SMA*(N_MANC/N_MA))):
        
                probN_MANC = (N_SMANC - NO_SMANC) / (N_MANC - NO_MANC)
        
                probD_MANC = np.nan_to_num(sp.binom(N_SMANC, NO_SMANC)) * \
                            np.nan_to_num(sp.binom(N_MANC-N_SMANC, NO_MANC-NO_SMANC)) / np.nan_to_num(sp.binom(N_MANC, NO_MANC))
    
            
                probP_MANC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMA/N_MA)*N_MANC))
                
                Tprob_MANC = probN_MANC * probD_MANC * probP_MANC
        
                prob_MANC = prob_MANC + Tprob_MANC 
            
    return prob_MANC

def probTR_MANC(N, N_S, N_M, N_MA, N_MANC, NO_SM, NO_SMA, NO_SMANC, NO_MANC):
    
    probTR_MANC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
     
        for N_SMA in range(NO_SMA, math.floor(N_SM*(N_MA/N_M))):
            
            
            for N_SMANC in range(NO_SMANC, math.floor(N_SMA*(N_MANC/N_MA))):
        
                probD_MANC = np.nan_to_num(sp.binom(N_SMANC, NO_SMANC)) * \
                            np.nan_to_num(sp.binom(N_MANC-N_SMANC, NO_MANC-NO_SMANC)) / np.nan_to_num(sp.binom(N_MANC, NO_MANC))
    
            
                probP_MANC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMA/N_MA)*N_MANC))
                
                TprobTR_MANC = probD_MANC * probP_MANC
        
                probTR_MANC = probTR_MANC + TprobTR_MANC 
            
    return probTR_MANC

# MCYC formula ------------------------------------------------------------------------------------------------------------------------

def prob_MCYC(N, N_S, N_M, N_MC, N_MCNC, N_MCYC, NO_SM, NO_SMC, NO_SMCYC, NO_MCYC):
    
    prob_MCYC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
            
        for N_SMC in range(math.floor(max(NO_SMC, N_SM*(N_MC/N_M))), N_MC):
           
            for N_SMCYC in range(math.floor(max(NO_SMCYC, N_SMC*(N_MCYC/N_MC))), N_MCYC):
        
               
                probN_MCYC = (N_SMCYC - NO_SMCYC) / (N_MCYC - NO_MCYC)
        
               
                probD_MCYC = np.nan_to_num(sp.binom(N_SMCYC, NO_SMCYC)) * \
                            np.nan_to_num(sp.binom(N_MCYC-N_SMCYC, NO_MCYC-NO_SMCYC)) / np.nan_to_num(sp.binom(N_MCYC, NO_MCYC))
    
            
                probP_MCYC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMC/N_MC)*N_MCNC))
                
                Tprob_MCYC = probN_MCYC * probD_MCYC * probP_MCYC
        
                prob_MCYC = prob_MCYC + Tprob_MCYC 
            
    return prob_MCYC

def probTR_MCYC(N, N_S, N_M, N_MC, N_MCNC, N_MCYC, NO_SM, NO_SMC, NO_SMCYC, NO_MCYC):
    
    probTR_MCYC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
            
        for N_SMC in range(math.floor(max(NO_SMC, N_SM*(N_MC/N_M))), N_MC):
           
            for N_SMCYC in range(math.floor(max(NO_SMCYC, N_SMC*(N_MCYC/N_MC))), N_MCYC):
        
                probD_MCYC = np.nan_to_num(sp.binom(N_SMCYC, NO_SMCYC)) * \
                            np.nan_to_num(sp.binom(N_MCYC-N_SMCYC, NO_MCYC-NO_SMCYC)) / np.nan_to_num(sp.binom(N_MCYC, NO_MCYC))
    
            
                probP_MCYC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMC/N_MC)*N_MCNC))
                
                TprobTR_MCYC = probD_MCYC * probP_MCYC
        
                probTR_MCYC = probTR_MCYC + TprobTR_MCYC 
            
    return probTR_MCYC

# MCNC formula ------------------------------------------------------------------------------------------------------------------------

def prob_MCNC(N, N_S, N_M, N_MC, N_MCNC, NO_SM, NO_SMC, NO_SMCNC, NO_MCNC):
    
    prob_MCNC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
        
        # To simplify the bounds on N_SMC, substituting the given values, NO_SMC = 23, NO_SMCNC/(N_MCNC/N_MC) = 15, so 
        # max(NO_SMC, N_SM*(N_MC/N_M)) <= N_SMC <= N_MC, and note the bounds always make sense

        for N_SMC in range(math.floor(max(NO_SMC, N_SM*(N_MC/N_M))), N_MC):
            
            
            for N_SMCNC in range(NO_SMCNC, math.floor(N_SMC*(N_MCNC/N_MC))):
          
                    probN_MCNC = (N_SMCNC - NO_SMCNC) / (N_MCNC - NO_MCNC)
        
                
                    probD_MCNC = np.nan_to_num(sp.binom(N_SMCNC, NO_SMCNC)) * \
                                np.nan_to_num(sp.binom(N_MCNC-N_SMCNC, NO_MCNC-NO_SMCNC)) / np.nan_to_num(sp.binom(N_MCNC, NO_MCNC))
    
            
                    probP_MCNC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMC/N_MC)*N_MCNC))
                
                    Tprob_MCNC = probN_MCNC * probD_MCNC * probP_MCNC
        
                    prob_MCNC = prob_MCNC + Tprob_MCNC 
            
    return prob_MCNC

def probTR_MCNC(N, N_S, N_M, N_MC, N_MCNC, NO_SM, NO_SMC, NO_SMCNC, NO_MCNC):
    
    probTR_MCNC = 0           
    
    for N_SM in range(NO_SM, math.floor(N_S*(N_M/N))):
   
        for N_SMC in range(math.floor(max(NO_SMC, N_SM*(N_MC/N_M))), N_MC):
            
            
            for N_SMCNC in range(NO_SMCNC, math.floor(N_SMC*(N_MCNC/N_MC))):
          
                    probD_MCNC = np.nan_to_num(sp.binom(N_SMCNC, NO_SMCNC)) * \
                                np.nan_to_num(sp.binom(N_MCNC-N_SMCNC, NO_MCNC-NO_SMCNC)) / np.nan_to_num(sp.binom(N_MCNC, NO_MCNC))
    
            
                    probP_MCNC = (1/((N_S/N)*N_M)) * (1/((N_SM/N_M)*N_MA)) * (1/((N_SMC/N_MC)*N_MCNC))
                
                    TprobTR_MCNC = probD_MCNC * probP_MCNC
        
                    probTR_MCNC = probTR_MCNC + TprobTR_MCNC 
            
    return probTR_MCNC