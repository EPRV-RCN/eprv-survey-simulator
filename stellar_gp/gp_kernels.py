
import numpy as np

# ===== Unit conversion helper functions =====
def omega_to_nu(omega):
    return omega / (2 * np.pi)

def nu_to_omega(nu):
    return 2 * np.pi * nu

def omega_sec_to_day(omega):
    return omega * 60 * 60 * 24

def omega_day_to_sec(omega):
    return omega / (60 * 60 * 24)

def S_sec_to_day(S):
    return S / (60 * 60 * 24)

def S_day_to_sec(S):
    return S * 60 * 60 * 24

def get_celerite_hypers(omega, S):
    tau = 2 * np.pi / omega
    sigma = np.sqrt(omega * S / np.sqrt(2))
    return tau, sigma

def get_S_omega_hypers(tau, sigma):
    omega = 2 * np.pi / tau
    S = sigma**2 / omega * np.sqrt(2)
    return omega, S


## NOW OSCILLAITION KERNEL

def SHO_latent(Delta_t, w,Q,S):
    Delta_t=np.abs(Delta_t)
    eta=np.sqrt( np.abs(1.0-1.0/(4.0*Q**2)) )
    return S * w * Q * np.exp(-w*Delta_t/(2.0*Q))*( np.cos(eta*w*Delta_t)+ 1.0/(2.0*eta*Q)*np.sin(eta*w*Delta_t) ) 


# def osc_psd(ω,w, Q, S)
#     @assert Q > 2
#     return 4 * S * w**4 / ((ω**2-w**2)**2 + w**2 * ω**2/Q**2) 
# end

def SHO_single_integral(Delta_t,delta,w,Q,S):
   
    eta=np.sqrt( np.abs(1.0-1.0/(4.0*Q**2)) ) 
    a=1/(eta*2*Q)
    
    y1=eta*w*((delta)/2 + Delta_t)
    y2=eta*w*(Delta_t - delta/2)
    
    def I_func(y):
        return np.exp(-a*y)*((1-a**2)*np.sin(y) - 2*a*np.cos(y))

    return S * w * Q * 1/(delta) * -1/(eta*w) * 1/(1+a**2) * (I_func(y2)-I_func(y1))



def SHO_single_integral_logic(Delta_t,delta,w,Q,S):
    #define our coordinate system
    #Delta_t must be positive
    #obs1 starts at t=0
    #obs 1 is the finite exposure, obs2 is instantaneous

    #these are the begin and end times for obs1
    #obs1 spans time p1 to p2
    Delta_t=abs(Delta_t)
    p1 = 0
    p2 = delta
    p3 = delta/2 + Delta_t

    int=0.0

    timescale = 1/(w/(2*np.pi))

    # Check if we need to use the instantaneous kernel
    # This will be true if the finite exposure is zero and/or
    # significantly less than the osc_timescale to be functionally zero
    if abs(delta/timescale) < 1e-4:
        return SHO_latent(Delta_t,w,Q,S) 

    if p2 <= p3:
        # "obs"2 did not occur during exposure of obs1
        # we can compute the left integral
        return SHO_single_integral(Delta_t,delta,w,Q,S) 
    elif p3 < p2:
        # "obs"2 did occur during exposure of obs1
        # we need to split the integral into a left and right side

        # Left side
        delta1new=p3
        Delta_tnew=delta1new/2
        int += SHO_single_integral(Delta_tnew,delta1new,w,Q,S)*delta1new
        
        # Right side
        delta1new = delta-p3
        Delta_tnew = delta1new/2
        int += SHO_single_integral(Delta_tnew,delta1new,w,Q,S)*delta1new

        return int/delta
    else:
        raise ValueError("You should not be here. Observation times are improperly defined")



def SHO_double_integral_separate(Delta_t,delta1,delta2,w,Q,S):
   
    eta=np.sqrt( np.abs(1.0-1.0/(4.0*Q**2)) ) 
    a=1/(eta*2*Q)
    
    y1=eta*w*((delta1+delta2)/2 + Delta_t)
    y2=eta*w*((delta1+delta2)/2 + Delta_t - delta1)
    y3=eta*w*((delta1-delta2)/2 + Delta_t)
    y4=eta*w*((delta1-delta2)/2 + Delta_t - delta1)
    
    def f1(y1,y2,a):
        return np.exp(-a*y2)*(a*np.sin(y2)+np.cos(y2)) - np.exp(-a*y1)*(a*np.sin(y1)+np.cos(y1)) 

    
    def f2(y1,y2,a):
        return np.exp(-a*y2)*(np.sin(y2)-a*np.cos(y2)) - np.exp(-a*y1)*(np.sin(y1)-a*np.cos(y1)) 
    
    # # I1 = 1/(eta*w) * 1/(1+a**2) * (1-a**2) * f1(y1,y2,a)
    # # I3 = 1/(eta*w) * 1/(1+a**2) * (1-a**2) * f1(y3,y4,a)
    
    # # I2 = -1/(eta*w) * 1/(1+a**2) * (2*a) * f2(y1,y2,a)
    # # I4 = -1/(eta*w) * 1/(1+a**2) * (2*a) * f2(y3,y4,a)

    # #return S * 1/(delta1*delta2) * 1/(eta*w) * 1/(1+a**2) * (I1-I2-I3+I4) 

    def I_plus(lower,upper):
        return 1/(eta*w) * 1/(1+a**2) * (1-a**2) * f1(lower,upper,a)
        

    def I_minus(lower,upper):
        return -1/(eta*w) * 1/(1+a**2) * (2*a) * f2(lower,upper,a)

    return S * w * Q * 1/(delta1*delta2) * 1/(eta*w) * 1/(1+a**2) * (I_plus(y1,y2)-I_minus(y1,y2)-I_plus(y3,y4)+I_minus(y3,y4))



def SHO_double_integral_overlap(delta,w,Q,S):
   
    eta=np.sqrt( np.abs(1.0-1.0/(4.0*Q**2)) ) 
    a=1/(eta*2*Q)
    
    y1=eta*w*delta
    y2=0.
    
    def f1(y1,y2,a):
        return np.exp(-a*y2)*(a*np.sin(y2)+np.cos(y2)) - np.exp(-a*y1)*(a*np.sin(y1)+np.cos(y1)) 
    
    def f2(y1,y2,a):
        return np.exp(-a*y2)*(np.sin(y2)-a*np.cos(y2)) - np.exp(-a*y1)*(np.sin(y1)-a*np.cos(y1)) 


    def I_plus(lower,upper):
        return 1/(eta*w) * 1/(1+a**2) * (1-a**2) * f1(lower,upper,a)

    def I_minus(lower,upper):
        return -1/(eta*w) * 1/(1+a**2) * (2*a) * f2(lower,upper,a)

    return S * w * Q * 1/(delta*delta) * 2 * 1/(eta*w) * 1/(1+a**2) * (I_plus(y1,y2) - I_minus(y1,y2) + 2*a*delta)


def SHO_kernel_full(Delta_t,delta1,delta2,w,Q,S):
    #define our coordinate system
    #Delta_t must be positive
    #obs1 starts at t=0
    #obs 1 is the longer exposure
    # if delta1 < delta2
    #     delta1temp=delta1
    #     delta1=delta2
    #     delta2=delta1temp
    # end

    delta1, delta2 = max(delta1, delta2), min(delta1, delta2)
    
    Delta_t=np.abs(Delta_t)
    
    #these are the begin and end times for both observations
    #obs1 spans time p1 to p2
    #obs2 spans time p3 to p4
    p1=0
    p2=delta1   
    p3=(delta1-delta2)/2+Delta_t
    p4=p3+delta2
    
    int=0.0

    timescale = 1/(w/(2*np.pi))
    
    # Check if we need to use the instantaneous kernel
    # This will be true if both exposures are zero and/or
    # significantly less than the osc_timescale to be functionally zero
    if (delta1 == delta2) and (np.abs(delta1/timescale) < 1e-8):
        return SHO_latent(Delta_t,w,Q,S) 
    elif delta2 == 0:
        # placeholder for single integral over delta
        # This really shouldn't happen but can come up when
        # doing GP conditioning and generating the "true" mean
        #delta2 = 1e-2 # choose something really small and continue with the double integral
        return SHO_single_integral_logic(Delta_t,delta1,w,Q,S)
    
    # We need to break the observations up into their parts
    if Delta_t == 0 and delta1==delta2:
        #obs1 and obs2 completely overlap
        return SHO_double_integral_overlap(delta1,w,Q,S) 
        
    elif Delta_t >= (delta1+delta2)/2:
        #obs1 and obs2 are completely separated
        return SHO_double_integral_separate(Delta_t,delta1,delta2,w,Q,S) 
        
    elif Delta_t < (delta1+delta2)/2 and p4 > p2:
        #obs 1 and obs 2 share some overlap
        # |----------|
        #        |-------|
        # p1    p3   p2 p4

        #break it up into 3 integrals        
        # Int 1
        # |------|
        #        |------|
        # p1    p3      p4
        delta1new=p3-p1
        delta2new=p4-p3
        Delta_tnew=delta1new/2 + delta2new/2
        int+=SHO_double_integral_separate(Delta_tnew,delta1new,delta2new,w,Q,S)*(delta1new*delta2new) 
        #we multiply by (delta1new*delta2new) to unnormalize so we can renormalize using the full 1/(d1*d2) at the end

        # Int 2
        #        |----|
        #             |---|
        #       p3   p2  p4
        delta1new=p2-p3
        delta2new=p4-p2
        Delta_tnew=delta1new/2+delta2new/2
        int+=SHO_double_integral_separate(Delta_tnew,delta1new,delta2new,w,Q,S)*(delta1new*delta2new)


        # Int 3 (overlap)
        #        |----|
        #        |----|
        #       p3   p2 
        deltaover=(p2-p3)
        int+=SHO_double_integral_overlap(deltaover,w,Q,S)*(deltaover**2)
        return int/(delta1*delta2) 

    elif Delta_t < (delta1+delta2)/2 and p4<=p2:
        # if obs 2 is completely within obs 1
        # |------------|
        #   |-------|
        # p1 p3    p4  p2
        print("WITHIN")
        
        # Int 1 (overlap)
        #   |-------|
        #   |-------|
        #  p3      p4 
        deltaover=delta2
        int+=SHO_double_integral_overlap(deltaover,w,Q,S)*(deltaover**2)
        
        # Int 2 (left)
        # |--|
        #    |------|
        # p1 p3    p4
        if p1 == p3:
            # this happens if exposures start at the exact same time
            # in this case, overlap above should cover the left integral, nothing to calculate
            int += 0
        else:
            delta1new=p3-p1
            delta2new=delta2
            Delta_tnew=abs(delta1new+(delta2-delta1new)/2)
            int+=SHO_double_integral_separate(Delta_tnew,delta1new,delta2new,w,Q,S)*(delta1new*delta2)
        
        # Int 3 (right) 
        #           |--|
        #    |------|
        #    p3    p4  p2
        if p2==p4:
            # this happens if exposures end at the exact same time
            # in this case, overlap above should cover the right integral, nothing to calculate
            int+=0
        else:
            delta1new=p2-p4
            delta2new=delta2
            Delta_tnew=delta1new/2. + delta2new/2
            int+=SHO_double_integral_separate(Delta_tnew,delta1new,delta2new,w,Q,S)*(delta1new*delta2)
    
        return int/(delta1*delta2)


## Quasi-periodic kernel
def qp(Delta_t,α,Γ,τ,P):
    return α**2 * np.exp(-0.5*Delta_t**2/τ**2 - Γ*np.sin(np.pi*Delta_t/P)**2)

## Squared exponential kernel
def sqexp(Delta_t,α,λ):
    return α**2 * np.exp(-0.5*Delta_t**2/λ**2)

## Periodic kernel
def per(Delta_t,α,ℓ,P):
    return α**2 * np.exp(-2/ℓ**2 * np.sin(np.pi*np.abs(Delta_t)/P)**2)

## Matern 3/2 kernel
def m32(Delta_t,σ,ρ):
    return σ**2 * (1+np.sqrt(3)*Delta_t/ρ)*np.exp(-np.sqrt(3)*Delta_t/ρ)

## Matern 5/2 kernel
def m52(Delta_t,σ,ρ):
    return σ**2 * (1+np.sqrt(5)*Delta_t/ρ + (5/3)*Delta_t**2/ρ**2)*np.exp(-np.sqrt(5)*Delta_t/ρ)

## Matern 5/2 kernel plus its first derivative
def m52pd(Delta_t,σ1,σ2,ρ):
    return σ1**2 * (1+np.sqrt(5)*Delta_t/ρ + (5/3)*Delta_t**2/ρ**2)*np.exp(-np.sqrt(5)*Delta_t/ρ) - σ2**2 * (-np.sqrt(5)/ρ)**2 /3 * np.exp(-np.sqrt(5) * Delta_t / ρ) * (1 + np.sqrt(5)*Delta_t/ρ - (np.sqrt(5)/ρ)**2 * Delta_t**2)
