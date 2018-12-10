"""
Created on Sat Dec  8 14:36:08 2018

@author: tony
"""
# Filename:
# python CHANG-ES_Pipeline_1.py

'''
IMPORT MODULES
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.io import fits
from scipy.stats import binned_statistic as bs
from matplotlib import ticker
from scipy.optimize import fmin

'''
DEFINE FUNCTIONS
'''

def Intensity_grad_map_mesh(I_arr):

    nx = len(I_arr[0,:])
    ny = len(I_arr[:,0])
   
    x = np.arange(1,nx-1,dtype=np.int)
    y = np.arange(1,ny-1,dtype=np.int)

    xx,yy=np.meshgrid(x,y, indexing='ij')
    
    dx = 0.5*(I_arr[yy,xx+1] - I_arr[yy,xx-1])
    dy = 0.5*(I_arr[yy+1,xx] - I_arr[yy-1,xx])
    
    gmap = np.zeros([ny,nx,2])

    gmap[yy, xx, 0] = np.sqrt(dx**2.0 + dy**2.0)
    gmap[yy, xx, 1] = np.arctan2(dy,dx)
   
    return gmap


def calc_disp(gal, pmap, cutval):

    disp0 = np.var(gal[gal == gal])
    disp1 = 0.0

    while(np.absolute((disp1-disp0)/disp0) > 0.005):
        
        disp1 = disp0
        disp0 = np.var(gal[(gal == gal) & (gal < cutval*np.sqrt(disp1))])
        dispp = np.var(pmap[(pmap == pmap) & (gal < cutval*np.sqrt(disp1))])

    return np.sqrt(disp0), np.sqrt(dispp*2.0/(4.0 - np.pi))


def shear_rot_est_corr(alphak, betak, xi_alpha):
       
    bkbk = np.zeros([3, 3])
  
    aplus = alphak  + betak
    aminus = alphak  - betak
  
    camb = np.mean(np.cos(2.0*(aminus))) / xi_alpha
    capb = np.mean(np.cos(2.0*(aplus))) / xi_alpha
    samb = np.mean(np.sin(2.0*(aminus))) / xi_alpha
    sapb = np.mean(np.sin(2.0*(aplus))) / xi_alpha
    ca = np.mean(np.cos(2.0*alphak ))/ xi_alpha 
    cb = np.mean(np.cos(2.0*betak)) 
    sa = np.mean(np.sin(2.0*alphak))/ xi_alpha
    sb = np.mean(np.sin(2.0*betak))
  
    deltabk = np.array([-samb, ca - cb, sa - sb])
  
    bkbk[0, 0] = 1.0 + camb
    bkbk[0, 1] = sa + sb
    bkbk[0, 2] = -ca - cb
    bkbk[1, 0] = bkbk[0, 1]
    bkbk[1, 1] = 1.0 - capb
    bkbk[1, 2] = -sapb
    bkbk[2, 0] = bkbk[0, 2]
    bkbk[2, 1] = bkbk[1, 2]
    bkbk[2, 2] = 1.0 + capb
    
    bkbk_inv = np.linalg.inv(bkbk)
    
    return np.dot(bkbk_inv, deltabk)


def cauchy(x0,*x):
    m, Circstd = x0
    data_x, data_y = Bedges, hist
    xtemp = np.linspace(-0.5*np.pi, 0.5*np.pi, num=1000)
    model = (np.sinh(2.0*Circstd**2))/(np.pi*(np.cosh(2.0*Circstd**2)-np.cos(2.0*(xtemp - m))))
    model = bs(xtemp, model, bins=data_x)[0]
    model = model/(np.sum(model)*(data_x[1]-data_x[0]))
    chisq = (((data_y-model)**2)).sum()
    
    return chisq

'''
DEFINE PARAMETERS
'''

band = 'C'
nbin = int(20)
res = vector_plot_resolution = int(3)

'''
NAME FILEPATH FOR .txt DATA, STOKES_I, P, ALPHA AND SAVED DATA PRODUCTS
'''

PATH = '/home/tony/Desktop/CHANG-ES/' 

'''
DEFINE NOISE LEVELS FOR CUTS
'''

cutvalIN =3 
cutvalPN = 3
cutvalI = 5 
cutvalP = 1 

'''
CREATE EMPTY LISTS
'''

MABC_shift = []
MEAN_ABC = []
ABstd_C = []
ABCREDCHI = []
WC = []
G1C = []
G2C = []
NGCC = []
SIGNALC = []

'''
READ IN DATA FOR C-BAND
'''

NGClist=np.loadtxt('%sCHANG-ES_Pipeline_1_data.txt' %PATH)
for row in NGClist:
    
    NGC	= int(row[0])		
    NGCC.append(NGC)				
    centroid = cent = int(row[1])		
    r =	int(row[2])		
    arcsecpixel = row[3]		
     
    '''
    USE THESE INPUTS TO DETERMINE FWHM AND K OF EVLA D-CONFIGURATION C-BAND BEAM
    '''
    
    nside = int((centroid*2)-1)    
    FWHM_beam_arcsec = 12.0
    FWHM_beam_PIX = FWHM_beam_arcsec / arcsecpixel   
    sigmaB = FWHM_beam_arcsec / (2.0*np.sqrt(2.0*np.log(2.0)))
    k_beam = 1/(2.0*sigmaB**2) 
       
    '''
    DEFINE ZOOM / MASK POSITION AND DIMENSIONS
    '''
    
    imin = centroid - r
    imax = centroid + r
    jmin = centroid - r
    jmax = centroid + r
    
    '''
    DATA PATH TO STOKES I MAPS
    '''
                          
    data_file_I = '%sN%d_D_%s_Rob0_Pbcor_Image.tt0.PBNEW.FITS' %(PATH,NGC,band) 
    input_gals = fits.open(data_file_I, memmap=True)
    gal = input_gals[0].data
    gal = np.array(gal[0,0,:,:])
    
    '''
    DATA PATH TO ALPHA MAPS, CONVERT TO RADIANS AND CONVERT FROM B-MODE TO E-MODE SPIN-2 LINEAR POLARISATION ANGLES 
    THE LAST LINE OF CODE IN THIS SECTION WRAPS THE SPIN-2 ANGLE BACK INTO THE -PI < ALPHA < PI REGIME
    '''
    
    path_to_alphas = '%sN%d_D_C_Rob0_X.FITS' %(PATH,NGC)
    input_gals = fits.open(path_to_alphas, memmap=True)
    alpha= input_gals[0].data
    alpha = np.array(alpha[0,0,:,:])
    alpha = np.radians(alpha)
    alpha = alpha + (np.pi/2)
    alpha = 0.5 * np.arctan2(np.sin(2*alpha), np.cos(2*alpha))  
    
    '''
    DATA PATH TO P MAPS (ALREADY CUT AT THE 3 SIGMA NOISE LEVEL)
    '''
    
    path_to_P = '%sN%d_D_C_Rob0_Pbcor_P.tt0.PBNEW.FITS' %(PATH,NGC)
    input_gals = fits.open(path_to_P, memmap=True)
    Pmap = input_gals[0].data
    Pmap = np.array(Pmap[0,0,:,:])
       
    '''
    USE STOKES I MAP TO FIND THE MAGNITUDE OF THE GRADIENT OF I AND BETA 
    (BETA = THE SPIN-2 ANGLE OF THE GRADIENT OF THE INTENSITY, IN REALITY THIS IS A STANDARD SPIN-1 VECTOR)
    '''
    
    gmap = Intensity_grad_map_mesh(gal)
    gradI = gmap[:,:,0]
    beta = gmap[:,:,1]
    
    '''
    USE ZOOM AREA TO BLANK OFF CENTRAL REGION OF I AND P MAPS TO FIND THE NOISE LEVEL AT 3 SIGMA IN THE REMAINING MAPS
    '''
                
    gal2 = np.copy(gal)
    gal2[jmin:jmax,imin:imax]=np.nan
    
    Pmap2 = np.copy(Pmap)
    Pmap2[jmin:jmax,imin:imax]=np.nan 

    dispI, dispP = calc_disp(gal2, Pmap2, cutvalIN)
    
    '''
    ZOOM IN TO ALL MAPS
    '''
    
    galZ = np.array(gal[jmin:jmax,imin:imax])
    PmapZ = np.array(Pmap[jmin:jmax,imin:imax])
    alphaZ = np.array(alpha[jmin:jmax,imin:imax])
    gradIZ = np.array(gradI[jmin:jmax,imin:imax])
    betaZ = np.array(beta[jmin:jmax,imin:imax])
    
    '''
    USE NOISE IN I MAP TO CUT GRADIENT AND DEFINE AN ARRAY 'used_pix' OF NP.NANs AND ONES
    '''
    
    used_gradI = np.ones_like(gradIZ )
    used_gradI[gradIZ  <= cutvalI*dispI] = np.nan        
    
    used_A = np.ones_like(alphaZ)
    cols, rows = used_A.shape
    for c in range(cols):
        for r in range(rows):
            used_A[c][r] = np.nan if np.isnan(alphaZ[c][r]) else used_A[c][r]
    
    '''
    CREATE A PLOT OF USED PIXELS SUPERIMPOSED OVER INVERTED GREYSCALE IMAGE OF GALAXY
    '''
    
    used_pix = np.multiply(used_gradI, used_A)
    
    plt.imshow(galZ, origin='lower', cmap='gist_yarg', alpha=1.0)
    plt.imshow(used_pix, origin='lower', cmap='autumn', alpha=0.25)
    plt.xlabel('RA (pixels)')
    plt.ylabel('dec (pixels)')
    plt.savefig('%s%d%s' %(PATH,NGC,band), bbox_inches='tight')
    plt.close('all')
    
    '''
    DEFINE NC = NUMBER OF USED PIXELS IN THE C-BAND ANALYSIS
    '''
    
    used_pix_flat = np.ravel(used_pix)
    used_pix_flat = used_pix_flat[used_pix_flat==used_pix_flat]
    NC = int(len(used_pix_flat))
    
    '''
    APPLY used_pix TO RELEVANT ZOOMED MAPS
    '''
    
    used_gal = np.multiply(galZ, used_pix)
    used_alpha = np.multiply(alphaZ, used_pix)
    used_beta = np.multiply(betaZ, used_pix)
    
    '''
    DETERMINE SIGNAL <a-b>
    '''
    
    abc_sig = used_alpha - used_beta
    abc_sig = 0.5 * np.arctan2(np.sin(2.0*abc_sig), np.cos(2.0*abc_sig))
    ABC_FLAT = np.ravel(abc_sig)
    ABC_FLAT = ABC_FLAT[ABC_FLAT ==ABC_FLAT]
    msin = (np.mean(np.sin(2.0*ABC_FLAT)))
    mcos = (np.mean(np.cos(2.0*ABC_FLAT)))
    mean_ABC = signalC= 0.5*np.arctan2(msin,mcos)
    SIGNALC.append(mean_ABC)
    r2 = (mcos**2.)+(msin**2)
    re2 = (NC/(NC-1.)*(r2-(1./NC))) 
    s2 = 0.25*np.log(1./re2)
    Circstd_ABC = np.sqrt(s2) 
       
    '''
    SWITCH TO DEDUCT <a-b> IF REQUIRED. THIS CAN HELP THE CS AND BIREF. ESTIMATOR FUNCTION OPTIMALLY
    '''
    
    #used_alpha = used_alpha- mean_ABC
    #used_alpha = 0.5*np.arctan2(np.sin(2.0*used_alpha), np.cos(2.0*used_alpha))
    
    '''
    ALPHA AND BETA MEANS
    '''
    
    a_flat = np.ravel(used_alpha)
    a_flat = a_flat[a_flat==a_flat]  
    msin = (np.mean(np.sin(2.0*a_flat)))
    mcos = (np.mean(np.cos(2.0*a_flat)))
    mean_A = 0.5*np.arctan2(msin,mcos)
    r2 = (mcos**2.)+(msin**2)
    re2 = (NC/(NC-1.)*(r2-(1./NC))) 
    s2 = 0.25*np.log(1./re2)
    circstd_A = np.sqrt(s2)
    
    
    b_flat = np.ravel(used_beta)
    b_flat = b_flat[b_flat==b_flat]  
    msin = (np.mean(np.sin(2.0*b_flat)))
    mcos = (np.mean(np.cos(2.0*b_flat)))
    mean_B = 0.5*np.arctan2(msin,mcos)
    r2 = (mcos**2.) + (msin**2)
    re2 = (NC/(NC-1.) * (r2-(1./NC))) 
    s2 = 0.25*np.log(1./re2)
    circstd_B = np.sqrt(s2) 
    
    '''
    RE-DETERMINE SIGNAL <a-b>
    '''
    
    abc_sig = used_alpha - used_beta
    abc_sig = 0.5 * np.arctan2(np.sin(2.0*abc_sig), np.cos(2.0*abc_sig))
    ABC_FLAT = np.ravel(abc_sig)
    ABC_FLAT = ABC_FLAT[ABC_FLAT ==ABC_FLAT]
    msin = (np.mean(np.sin(2.0*ABC_FLAT)))
    mcos = (np.mean(np.cos(2.0*ABC_FLAT)))
    mean_ABC = signalC= 0.5*np.arctan2(msin,mcos)
    r2 = (mcos**2.)+(msin**2)
    re2 = (NC/(NC-1.)*(r2-(1./NC))) 
    s2 = 0.25*np.log(1./re2)
    circstd_ABC = np.sqrt(s2) 
        
    abc = a_flat - b_flat
    abc = 0.5 * np.arctan2(np.sin(2.0*abc), np.cos(2.0*abc))
    
    msin=(np.mean(np.sin(2.0*abc)))
    mcos=(np.mean(np.cos(2.0*abc)))
    mean_AB_shift=0.5*np.arctan2(msin,mcos)
    MABC_shift.append(mean_AB_shift)
        
    '''
    CREATE AND SAVE ALPHA & BETA HISTOGRAM
    '''
    
    plt.hist(np.degrees(a_flat),range=[-90.0,90.0], bins=nbin, edgecolor='red', linewidth=1.1, histtype='step', normed=True, label=r'$alpha$')	
    plt.hist(np.degrees(b_flat),range=[-90.0,90.0], bins=nbin, edgecolor='blue', facecolor='b', linewidth=1.1, histtype='stepfilled', alpha=0.3, normed=True, label=r'$beta$')
    plt.xlabel(r' $alpha$'' (degs) or 'r'$beta$'' (degs)')
    plt.ylabel(r' $P(alpha)$'' or 'r' $P(beta)$')
    plt.legend(loc=2)
    plt.savefig('%s%d%s_A.png' %(PATH,NGC,band), bbox_inches='tight')
    plt.close('all')
        
    '''
    CREATE AND SAVE ALPHA - BETA HISTOGRAM C-BAND
    '''
    
    abc_flat = np.copy(abc)       
    msin=(np.mean(np.sin(2.0*abc_flat)))
    mcos=(np.mean(np.cos(2.0*abc_flat)))
    mean_abc = 0.5*np.arctan2(msin,mcos)
    MEAN_ABC.append(mean_abc)
    r2 = (mcos**2)+(msin**2)
    re2 = (NC/(NC-1)*(r2-(1/NC))) 
    s2 = 0.25*np.log(1/re2)
    ABCircstdC=np.sqrt(s2)
    ABstd_C.append(ABCircstdC)
    hist, Bedges = np.histogram(abc_flat,range=[-0.5*np.pi,0.5*np.pi], bins=nbin, normed=True)
    nbar = np.mean(hist)
    reduchisqC=(1/(np.float(nbar)*len(hist)))*np.sum((hist-nbar)**2) 
    ABCREDCHI.append(reduchisqC)
    
    '''
    CREATE AND SAVE COMPOSITE PLOT
    '''
    
    fig = plt.figure(frameon=True)        
    cmap = colors.ListedColormap(['red', 'green', 'blue', 'blue', 'green', 'red'])                                  
    bounds = [-90, -60,-30, 0.0, 30, 60, 90]
    normal = colors.BoundaryNorm(bounds, cmap.N)
    im1 = plt.imshow(np.degrees(abc_sig), origin='lower',cmap=cmap, alpha=0.7, extent=[0,(r/3),0,(r/3)])
    cbar = fig.colorbar(im1, boundaries = bounds, norm = normal, cmap=cmap)           
    plt.clim(-90,90)
    im2 = plt.contour(galZ, origin='lower', extent=[0,(r/res),0,(r/res)], cmap='gist_gray', alpha=0.7)     
    U = np.cos(used_alpha)
    V = np.sin(used_alpha)                      
    im3 = plt.quiver(U[::res, ::res], V[::res, ::res], pivot='mid', headwidth=0, headlength=0, headaxislength=0, scale=30)    
    im4 = plt.contour(galZ, origin='lower', extent=[0,(r/res),0,(r/res)], locator=ticker.LogLocator(), cmap='gist_gray', alpha=0.7)   
    circle1 = plt.Circle(((FWHM_beam_PIX/(2*res))+1, (FWHM_beam_PIX/(2*res))+1), (FWHM_beam_PIX/(2*res)), color='r', fill=True, linewidth = 0.5)
    plt.gcf().gca().add_artist(circle1)    
    plt.xlabel('RA (pixels)')
    plt.ylabel('dec (pixels)')
    plt.savefig('%s%d%sVectorPlot.png' %(PATH,NGC,band), bbox_inches='tight')
    plt.close('all')
    
    '''
    BETA VECTOR PLOT
    '''
    
    fig = plt.figure(frameon=True)     
    im1 = plt.imshow(galZ, origin='lower', extent=[0,(r/3),0,(r/3)], alpha=1.0, cmap='gist_yarg') #,vmin=galmin,vmax=galmax
    U = np.cos(used_beta)
    V = np.sin(used_beta)                      
    im2 = plt.quiver(U[::3, ::3], V[::3, ::3], pivot='mid',headwidth=0,headlength=0,headaxislength=0,scale=30, color='red')
    plt.xlabel('RA (pixels)')
    plt.ylabel('dec (pixels)')
    plt.savefig('%s%d%sbeta.png' %(PATH,NGC,band),bbox_inches='tight')  
    plt.close('all')
    
    '''
    USE FMIN TO FIT WRAPPED CAUCHY DISTRIBUTION TO HISTOGRAM AND SAVE PLOT
    '''

    data_x, data_y = Bedges, hist   
    guess = [mean_abc, ABCircstdC]
    best_parameters = fmin(cauchy, guess, (data_x,data_y))   
    m, Circstd = best_parameters[0], best_parameters[1]    
    xtemp = np.linspace(-0.5*np.pi, 0.5*np.pi, num=1000)
    model = (np.sinh(2.0*Circstd**2))/(np.pi*(np.cosh(2.0*Circstd**2)-np.cos(2.0*(xtemp - m))))
    fit = bs(xtemp, model, bins=data_x)[0]
    fit = fit/(np.sum(fit)*(data_x[1]-data_x[0]))
    Bedges = np.array([0.5*(Bedges[i] + Bedges[i+1]) for i in np.arange(len(Bedges)-1)])

    plt.hist(abc_flat,range=[-0.5*np.pi, 0.5*np.pi], bins=nbin, edgecolor='black', linewidth=1.1, histtype='step', normed=True)	
    plt.xlabel(r' $alpha - beta$'' (degs)')
    plt.ylabel(r' $P(alpha - beta)$')    
    plt.plot(Bedges, fit, 'r--', alpha=1.0, linewidth=0.8)    
    plt.savefig('%s%d%s_A-B.png' %(PATH,NGC,band), bbox_inches='tight')
    plt.close('all')
    
    '''
    DETERMINE XI_APLHA CORRECTING TERM BASED ON THE FITTED WRAPPED CAUCHY DISPERSION (ASTROPHYSICAL SCATTER)
    '''
          
    xi_alpha = np.exp(-2.0 * Circstd**2)    

    '''
    INPUT THE C-BAND ALPHA & BETA ARRAYS INTO THE COSMIC SHEAR AND BIREFRINGENCE ESTIMATOR ALONG WITH THE XI_APLHA CORRECTING TERM
    '''
    
    w, g1, g2 = shear_rot_est_corr(a_flat, b_flat, xi_alpha)
    
    WC.append(w)    
    G1C.append(g1)   
    G2C.append(g2)
    
    print 'NGC%d omega = %1.6f gamma 1 = %1.6f gamma 2 = %1.6f' %(NGC,w,g1,g2) 
    
    
    
    
   