#!/usr/bin/env python

"""
Module with functions to determine the required values to plot the median and mean curves
for two variables.

Writer: Lucas J. Zenocratti
2021, GalaxiesModels Group

"""

# ======================================================================================================================= 

import numpy as np
from scipy import stats

# ======================================================================================================================= 

def median_curve(x,y,bins=10,min_count=10,perc_low=25,perc_high=75):
    """
    Determine median values of 'x' and 'y', dividing the 'x' interval in 'bins' number of bins. The
    lower and higher errors computed for 'x' and 'y' correspond to the respective 'perc_low' and
    'perc_high' percentiles in each bin. This function returns median values and errors only in bins
    where the number of elements is at least 'min_count'.
    
    Parameters:
            x,y: variables to use to construct the median curve. Both must have the same length.
            bins: number of bins of variable x to use (default: 10).
            min_count: minimum number of x values in each bin. For a given x bin, if the number of
            elements is lower than min_count, then that bin will not be considered (default: 10).
            perc_low: percentile to use to define a lower bound of error in each bin, for both 
            x and y variables (default: 25).  
            perc_high: percentile to use to define a higher bound of error in each bin, for both 
            x and y variables (default: 75).   
    
    Returns:
            median_plot: plot of the median curve.
            xmedian: median values of x variable in each x bin.
            ymedian: median values of y variable in each x bin.
            xerror: array of x errors. It is the array [[xerr1_low, ... , xerrN_low],[xerr1_high, ... , xerrN_high]], where
            N is the number of bins.
            yerror: array of y errors. It is the array [[yerr1_low, ... , yerrN_low],[yerr1_high, ... , yerrN_high]], where
            N is the number of bins.
            elements: number of elements in each bin.
    
    """

    # Define the median values of variable y, the edges of bins and the binnumber assigned to each value.
    ymedian,bin_edges,binnumber=stats.binned_statistic(x,y,statistic='median',bins=bins)
    
    # Initialize arrays for median values of variable x, and x and y errors
    xmedian=np.ones_like(ymedian)
    
    xerr_low=np.ones(len(bin_edges)-1,dtype=float)
    xerr_high=np.ones(len(bin_edges)-1,dtype=float)
    
    yerr_low=np.ones(len(bin_edges)-1,dtype=float)
    yerr_high=np.ones(len(bin_edges)-1,dtype=float)
    
    elements=np.ones_like(ymedian)    
    
    # For each bin, search elements corresponding to that bin
    for k in range(bins):
        mask=np.where(binnumber==k+1)[0]
        
        # Assign median and errors equal to -99 if there are not enough elements in bin
        if (len(x[mask])<min_count):
            xmedian[k]=-99
            xerr_low[k]=-99
            xerr_high[k]=-99
        
            ymedian[k]=-99
            yerr_low[k]=-99
            yerr_high[k]=-99
            
            elements[k]=0
            
        # Calculate x median and errors in the bin, if there are enough elements in bin     
        else:
            xmedian[k]=np.median(x[mask])
            xerr_low[k]=xmedian[k]-np.percentile(x[mask],perc_low)
            xerr_high[k]=np.percentile(x[mask],perc_high)-xmedian[k]
        
            yerr_low[k]=ymedian[k]-np.percentile(y[mask],perc_low)
            yerr_high[k]=np.percentile(y[mask],perc_high)-ymedian[k]
            
            elements[k]=len(x[mask])
    
    # Identify bins with xmedian and ymedian not equal to -99
    mask_ok=np.where((xmedian != -99) & (ymedian != -99))[0]
    
    xmedian=xmedian[mask_ok]
    ymedian=ymedian[mask_ok]
    xerr_low=xerr_low[mask_ok]
    xerr_high=xerr_high[mask_ok]
    yerr_low=yerr_low[mask_ok]
    yerr_high=yerr_high[mask_ok]
    elements=elements[mask_ok]
    
    # Define arrays of x and y errors
    xerror=np.array([xerr_low,xerr_high])    
    yerror=np.array([yerr_low,yerr_high])
    
    return xmedian,ymedian,xerror,yerror,elements
    
# =======================================================================================================================   
    
def mean_curve(x,y,bins=10,min_count=10):
                      
    """
    Determine mean values of 'x' and 'y', dividing the 'x' interval in 'bins' number of bins. The 
    errors computed for 'x' and 'y' correspond to the respective 1-sigma in each bin. This function 
    returns mean values and errors only in bins where the number of elements is at least 'min_count'.
    
    Parameters:
            x,y: variables to use to construct the mean curve. Both must have the same length.
            bins: number of bins of variable x to use (default: 10).
            min_count: minimum number of x values in each bin. For a given x bin, if the number of
            elements is lower than min_count, then that bin will not be considered (default: 10).
    
    Returns:
            xmean: mean values of x variable in each x bin.
            ymean: mean values of y variable in each x bin.
            xerror: 1-sigma of x variable in each bin.
            yerror: 1-sigma of y variable in each bin.
            elements: number of elements in each bin.
    
    """                      

    # Define the mean values of variable y, the edges of bins and the binnumber assigned to each value.
    ymean,bin_edges,binnumber=stats.binned_statistic(x,y,statistic='mean',bins=bins)
    
    # Initialize arrays for mean values of variable x, and x and y errors
    xmean=np.ones_like(ymean)
    
    xerr=np.ones(len(bin_edges)-1,dtype=float)       
    yerr=np.ones(len(bin_edges)-1,dtype=float)
    
    elements=np.ones_like(ymean)
        
    # For each bin, search elements corresponding to that bin    
    for k in range(bins):
        mask=np.where(binnumber==k+1)[0]
        
        # Assign mean and errors equal to -99 if there are not enough elements in bin
        if (len(x[mask])<min_count):
            xmean[k]=-99
            xerr[k]=-99
                   
            ymean[k]=-99
            yerr[k]=-99
            
            elements[k]=0
                        
        # Calculate x mean and errors in the bin, if there are enough elements in bin     
        else:
            xmean[k]=np.mean(x[mask])
            xerr[k]=np.std(x[mask])
                
            yerr[k]=np.std(y[mask])
            
            elements[k]=len(x[mask])
            
    xerror=xerr   
    yerror=yerr
    
    # Identify bins with xmean and ymean not equal to -99
    mask_ok=np.where((xmean != -99) & (ymean != -99))[0]
    
    xmean=xmean[mask_ok]
    ymean=ymean[mask_ok]
    xerror=xerror[mask_ok]
    yerror=yerror[mask_ok]    
    elements=elements[mask_ok]
                                     
    return xmean,ymean,xerror,yerror,elements


# ======================================================================================================================= 

# EoM