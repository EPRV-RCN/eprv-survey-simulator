import numpy as np
from scipy.interpolate import interp1d
from .gp_kernels import nu_to_omega

def get_bolo_corr(teff):
    teff_to_interp=[4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000]
    corr_to_interp=[1.4270115478150247,1.3077233084291158,1.2087260623207006,1.1249672368175945,1.0531232535874526,
    0.9908608226873058,0.9364574118651322,0.8885935199046348,0.8462310783980914,0.8085374670815352,0.7748351505122221]
    
    bolo_itp = interp1d(teff_to_interp, corr_to_interp, kind='linear')
    
    bolocorr_atTeff_kepler=bolo_itp(teff)
    
    return bolocorr_atTeff_kepler


def calc_Pg(logg, teff, nu_max_sun=3100, teff_sun=5777, logg_sun=4.43, amax_sun_rv=0.19, cenv_sun=331, deltanu_sun=134.9):
    
    nu_max = calc_nu_max(logg,teff)
    
    cenv = 0.174*nu_max**0.88
    
    agran = 3382*nu_max**-0.609
    
    Pg_phot = np.sqrt(1./(2*np.pi*cenv**2))*(agran/4.57)**(2./0.855)
    Pg_phot=(3335.0/4.57*nu_max**(-0.564))**(2.0/0.855)*(1.0/cenv)*(1.0/np.sqrt(2.0*np.pi))
    
    beta = get_bolo_corr(teff)
    
    r_osc = 20*beta*(teff/teff_sun)**(-0.95)
    
    return Pg_phot/r_osc**2

def calc_a_rv(logg, teff, teff_sun=5777, logg_sun=4.43):
    agran=3382*calc_nu_max(logg,teff)**-0.609
    rgran=100.*(teff/teff_sun)**(-32./9.)*(10**logg/10**logg_sun)**(2./9.)
    
    return agran/rgran



def calc_nu_max(logg, Teff):
    # Solar reference values
    logg_sun = 4.43
    Teff_sun = 5777
    nu_max_sun = 3090  # μHz
    return nu_max_sun * 10**(logg - logg_sun) * (Teff / Teff_sun)**-0.5


## Stellar scaling functions
def get_stellar_hypers(logg=4.43,Teff=5777):
    nu_max = calc_nu_max(logg,Teff)

    # Oscillation parameters
    Q = 10**(0.4862+0.1135* np.log10(nu_max))
    nu_osc = nu_max*1e-6
    wosc = nu_to_omega(nu_osc)

    Pg = calc_Pg(logg,Teff)
    Sosc = Pg * 1e6 /(4 * Q**2) 


    a_rv=calc_a_rv(logg,Teff)
    b1=0.317*(nu_max)**(0.97)
    b2=0.948*nu_max**(0.992)

    w1=2*np.pi*(b1)*1e-6
    w2=2*np.pi*(b2)*1e-6

    S1 = a_rv**2/(np.sqrt(2)*w1/2)
    S2 = a_rv**2/(np.sqrt(2)*w2/2)

    return wosc, Sosc, Q, w1, S1, w2, S2