import numpy as np
import eagleSqlTools as sql
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.cosmology import Planck15 as cosmo

def retrieve_ids(usr,pwd,simu,snap,galid):
    """
    Function to download from EAGLE database the galaxy identifiers needed to construct the merger tree
    of a given galaxy. The function connects to EAGLE database by means of eagleSqlTools, and it requires
    an user name and password to connect.
    
    The arguments of the function are database user name and password (both must be a string variable), 
    simulation to use (string variable to), snapnum at which the galaxy is (integer variable), 
    and its GalaxyID (integer variable). , 
    and returns a dictionary with the identifiers GalaxyID, TopLeafID, LastProgID, and DescendantID 
    (all of them as integer variables).
    
    Parameters:
        usr: user name to connect to EAGLE database (str).
        pwd: password of the user name to connect to EAGLE database (str).
        simu: simulation from which to download data (str).
        snap: SnapNum at which is located the galaxy whose data are required (int).
        galid: GalaxyID of required galaxy (int).
        
    Returns:
        data: dictionary with all the identifiers required to build the merger tree
              (GalaxyID, LastProgID, TopLeafID, and DescendantID). All of these identifiers 
              are of type int64.
        
    """

    # Query to the database
    con = sql.connect(usr, password=pwd)

    query = "SELECT \
             sub.GalaxyID, sub.TopLeafID, sub.LastProgID, sub.DescendantID \
             FROM "+simu+"_subhalo as sub \
             WHERE sub.SnapNum="+str(snap)+" and sub.GalaxyID="+str(galid)

    # Execute query 
    exquery = sql.execute_query(con, query)
    
    # List of column names of downloaded data
    colnames=(exquery.view(np.recarray).dtype.names)
    
    
    # Dictionary of identifiers
    data={}
    for name in colnames:
        data[name]=exquery[name].item()

    return data
    
# ================================================================================================================    
# ================================================================================================================    
    
def download_merger_tree(usr,pwd,simu,galid,lastprogid,table='SubHalo',table_alias='sub',columns='All'):

    """
    Function to download the full merger tree of an EAGLE galaxy. The function needs the GalaxyID and LastProgID of the
    required galaxy. Data from a specifically given table of EAGLE database can be downloaded, althougth the function will
    download the variable GalaxyID from the SubHalo table (which is used to merge different EAGLE galaxies tables), and 
    it will called 'SubHaloGalaxyID'. Therefore, note that 'GalaxyID'='SubHaloGalaxyID'. Also optionally, the
    argument it is possible to download only a given set of columns from database (it defaults
    to 'All' in order to download all the columns from the desired table). 
    
    Parameters:
        usr: user name to connect to EAGLE database (str).
        pwd: password of the user name to connect to EAGLE database (str).
        simu: simulation from which to download data (str).
        galid: GalaxyID of required galaxy (int).
        lastprogid: LastProgID corresponding to the desired galaxy (int).
        table: table of EAGLE database from which to download data (str). Default: 'SubHalo'.
        table_alias: alias of table from which to download data (str). Default: 'sub'.
        columns: columns to download from the required table. It can be a list of str with column
                 names, or the str 'All' to download all the columns of the table. Default: 'All'.
                 
    Returns:
        prog_table: table (dictionary) with the required data of the merger tree.
                           
    NOTE: if only the main branch of the merger tree needs to be downloaded, when calling this function the argument 
    'lastprogid' must be set with the TopLeafID identifier.  
    
    """
    
    # Data to download
    if columns=='All':
        auxcol=table_alias+'.*'
    elif columns!='All':
        auxcol=[table_alias+'.'+item for item in columns]
        auxcol=",".join(auxcol)

    # Query to the database
    query = (
             "SELECT "+auxcol+',subaux.GalaxyID as SubHaloGalaxyID'+
             " FROM " +simu+"_"+table+" as "+table_alias+","+simu+"_subhalo as subaux"+
             " WHERE subaux.GalaxyID>="+str(galid)+" and subaux.GalaxyID<="+str(lastprogid)+
             " and subaux.GalaxyID="+table_alias+".GalaxyID"         
            )
             
    con = sql.connect(usr, password=pwd)
    
    # Execute query 
    exquery = sql.execute_query(con, query)
    
    # List of column names of downloaded data
    colnames=(exquery.view(np.recarray).dtype.names)
    
    # Dictionary of downloaded data
    prog_table={}
    for name in colnames:
        prog_table[name]=exquery[name]

    return prog_table

# ================================================================================================================    
# ================================================================================================================   

def merger_tree(data,galid,lastprogid,topid):
    """
    Function to construct the merger tree of a given galaxy, using a downloaded data file. The data file must
    contain at least data of the main progenitors of the required galaxy.
    
    Parameters:
        data: table to use to construct the merger tree. It should be a dictionary-like object.
        galid: GalaxyID of galaxy whose merger tree will be constructed (int).
        lastprogid: LastProgID corresponding to the desired galaxy (int).
        topid: TopLeafID corresponding to the desired galaxy (int).
        
    Returns:
        raw_tree: dictionary containing data of the galaxies in the merger tree.
    
    """
    
    # Identify galaxies in the tree of required galaxy.
    mask=(data['GalaxyID']>=galid) & (data['GalaxyID']<=lastprogid)
    raw_tree={}
    for key in data.keys():
        raw_tree[key]=data[key][mask]
        
    # Order galaxies according to descendant SnapNum  
    mask_order=(-1*raw_tree['SnapNum']).argsort()
    for key in raw_tree.keys():
        raw_tree[key]=raw_tree[key][mask_order] 
        
    # Flag to indicate if galaxy belongs to main branch
    raw_tree['Flag_main']=np.full(len(raw_tree['GalaxyID']),False)    
    # Mask with galaxies of the main branch
    mask_main=np.where((raw_tree['GalaxyID']>=galid) & (raw_tree['GalaxyID']<=topid))[0]
    raw_tree['Flag_main'][mask_main]=True
    
    # Number of progenitors of each galaxy in the tree
    raw_tree['Num_Prog']=np.ones_like(raw_tree['GalaxyID'])
    for k in range(len(raw_tree['GalaxyID'])):
        mask_prog=np.where(raw_tree['DescendantID']==raw_tree['GalaxyID'][k])[0]
        if len(mask_prog)==0:
            raw_tree['Num_Prog'][k]=0
        else:
            raw_tree['Num_Prog'][k]=len(mask_prog)
        
    return raw_tree

# ================================================================================================================    
# ================================================================================================================   

def main_branch(tree,galid,topid):
    """
    Function to identify only main progenitors of a given galaxy in its merger tree.
    
    Parameters:
        tree: data (dict-like) of all galaxies in the merger tree of the desired galaxy.
        galid: GalaxyID of the desired galaxy (int).
        topid: TopLeafID corresponding to the desired galaxy (int).
        
    Returns:
        main_branch: dictionary with data of the corresponding main progenitors of the desired galaxy.
    
    """
    
    mask_main=(tree['GalaxyID']>=galid) & (tree['GalaxyID']<=topid)
    main_branch={}
    for key in tree.keys():
            main_branch[key]=tree[key][mask_main]
    
    return main_branch

# ================================================================================================================    
# ================================================================================================================   

def plot_merger_tree(in_tree,galaxyID,lastprogID,topleafID,full_tree=False,
                     yvar='Redshift',invert_yaxis=True,
                     color_var='Stars_Mass',log_colorvar=True,color_var_label=r'$\log(M_\bigstar)$',
                     cmap='jet',plot_bad_color=False,bad_color='k'):                     
                     
    """
    Function to plot merger tree of a given EAGLE galaxy. This function needs as an argument the merger tree
    of the galaxy, and every galaxy of it must have at least the corresponding Redshift, SnapNum, DescendantID,
    and Stars_Mass data, in order to plot the merger tree. By default, the function only plots the main branch
    of the tree, but the galaxies that merge with the main branch and their corresponding (sub) main branchs 
    can be plotted, setting 'full_tree=True'. Also by default, the function will plot the tree using the Redshift
    variable as 'yvar', but another time measurements (e.g. SnapNum or cosmic time) may be chosen, depending on 
    the data available in the tree. The symbols in the plot are colour-coded acording to the variable 'color_var'
    (by default, it is set as the stellar mass), in logarithmic scale if desired. 
    
    Parameters:
        in_tree: input tree (dict-like) to plot.
        galaxyID: GalaxyID corresponding to the galaxy which merger tree will be plotted (int).
        lastprogID: LastProgID corresponding to the GalaxyID required (int).
        topleafID: TopLeafID corresponding to the GalaxyID required (int).
        full_tree: logical variable to plot only the main branch of the merger tree (full_tree=False), or to 
                   plot the main branch and the galaxies that merge with it togeter with their (sub) main 
                   branches (full_tree=True). Default: False.
        yvar: name of variable to use in the y-axis of the plot. It must be related to time measurement, such
              as 'SnapNum', 'Redshift', 'CosmicTime', 'LookBackTime', etc. Default: 'Redshift'.
        invert_yaxis: if True, invert the y-axis in the plot. Default: True.
        color_var: name of variable to use to color-code the symbols in the plot. This str must be included in
                   the keys of the in_tree. Default: 'Stars_Mass'.
        log_colorvar: if True, use log scale to color-code the symbols of the plot.
        color_var_label: label that identifies the variable used to color-code the symbols of the plot. This str
                         will be used as the label of the color reference bar in the plot. Default: r'$\log(M_\bigstar)$'. 
        cmap: name of color map used to color-code the symbols in the plot. Default: 'jet'.
        plot_bad_color: if True, plot galaxies in the tree with 'bad values' of log(color_var). Default: False
        bad_color: color to use to identify 'bad values' of variable used to color-code when coding with logscale
                   (for example, galaxies with color_var=0 and hence with log(color_var)=-inf). This color should
                   not be included in the cmap used, to avoid confusion. This parameter defaults to 'k', but it is
                   irrelevant if plot_bad_color=False.
                     
    Returns:
        fig0,ax0: figure and axis objects that contain and define the plot.
    
    """
    # Dictionary to store the galaxies in the tree
    tree={}
    
    # Order input tree according to descendant GalaxyID
    mask_order=(-1*in_tree['GalaxyID']).argsort()
    for key in in_tree.keys():
        tree[key]=in_tree[key][mask_order]
    
    # Number of progenitors of each galaxy in the tree
    num_prog=np.ones_like(tree['GalaxyID'])
    for k in range(len(tree['GalaxyID'])):
        mask_prog=np.where(tree['DescendantID']==tree['GalaxyID'][k])[0]
        if len(mask_prog)==0:
            num_prog[k]=1
        else:
            num_prog[k]=len(mask_prog)    
        
    # Identify the main branch of the tree
    tree_main_branch={}
    mask_main=np.logical_and(tree['GalaxyID']>=galaxyID,
                                 tree['GalaxyID']<=topleafID) 

    for key in tree.keys():
        tree_main_branch[key]=tree[key][mask_main]
    num_prog_main=num_prog[mask_main]      
        
    # Variable used to color-code
    if log_colorvar==True:        
        cvar=np.log10(tree[color_var])   # NOTA: si no se puede calcluar log10(variable) porque
                                         # la variable es nula por ejemplo, la función va a tirar
                                         # un warning...
    else:
        cvar=tree[color_var]
        
    #--------------------------------------------------------------------------
    # Esto es en caso de que se tome log y la variable para colorear sea cero
    # En estos casos, se asigna el color color_under (con el 'set_under' anterior).
    mask_color=~(np.logical_or(np.isinf(cvar),np.isnan(cvar)))
    #--------------------------------------------------------------------------
    
    # Set up the colours
    cmap=plt.get_cmap(cmap)
    if plot_bad_color==True:
        cmap.set_under(bad_color)  # Color asignado para los "valores malos" de la variable a usar para colorear.         
    
    # Limits of variable used to color-code. Modify if necessary.
    vmin=min(cvar[mask_color])
    vmax=max(cvar[mask_color])
      
    # Assign colors    
    a=1./float(vmax-vmin)
    b=-a*vmin
    cs=(a*cvar+b)
    color_plot=cmap(cs)
    color_plot_main=color_plot[mask_main]
    tree['color_plot']=color_plot
    
    if plot_bad_color==False:
        for key in tree.keys():
            tree[key]=tree[key][mask_color]
            
    
    # Define figure and axis to plot
    fig0,ax0=plt.subplots(figsize=(25,10))   

    # Auxiliary varibales for Main branch
    galid_main=tree_main_branch['GalaxyID']
    desc_main=tree_main_branch['DescendantID']
    num_merge_main=num_prog_main-1
    yplot_main=tree_main_branch[yvar]     # y-coordinates to plot galaxies in main branch
    # x-coordinate to begin the plot
    xbegin=0 
    xplot_main=xbegin*np.zeros_like(yplot_main)   # x-coordinates to plot galaxies in main branch.
                                                  # Asigno ceros, pero luego se modifican
    
    # Auxiliar x-coordinates to determine x-coordinates of progenitors
    xcoor_aux=xbegin
    xcoor_aux2=xbegin
    
    # Coordinates to plot secondary progenitors. Inician vacías.
    xplot_prog=[]
    yplot_prog=[]
    
    # Recorro las galaxias de la main branch para identificar sus progenitores
    for gal,ycoor,color,l in zip(galid_main,yplot_main,color_plot_main,range(len(yplot_main))):   
         
        # Look for galaxies that descend to the main branch galaxy
        mask_prog=np.where((tree['DescendantID']==gal))[0] 
        
        # En caso de tener más de un progenitor (i.e, main progenitor+secondary progenitors)
        if (len(mask_prog)>1): 
            j=0    # Esto es un contador que se usará para "separar" horizontalmente los progenitores
            # Recorro cada progenitor que desciende a la galaxia de la main branch
            for k in mask_prog:
                # Si el progenitor es el principal
                if (tree['Flag_main'][k])==True: 
                    xplot=xplot_main[l]
                    yplot=yplot_main[l]
                
                # Para los progenitores secundarios:
                else:                  
                    xplot=xbegin-5*(j+1)+xcoor_aux
                    yplot=yplot_main[l-1]
                    #ax0.scatter(xplot,yplot,s=50,c=[color_plot[k]])
                                   
                    j=j+1
                    
                    xcoor_aux2=xplot-3
                    xplot_prog=np.append(xplot_prog,xplot)
                    yplot_prog=np.append(yplot_prog,yplot)
                    
                    # Identify secondary progenitors' progenitor tree
                    sub_tree=merger_tree(tree,tree['GalaxyID'][k],tree['LastProgID'][k],tree['TopLeafID'][k])
                    
                    # Solo se grafica la main branch de los progenitores secundarios
                    sub_tree_mask_main=sub_tree['Flag_main']==True
                    sub_tree_main={}
                    for key in sub_tree.keys():
                        sub_tree_main[key]=sub_tree[key][sub_tree_mask_main]
                    
                    xcoord_main=xplot
                    ycoord_main=sub_tree_main[yvar]
                    
                    # Plot secondary main branches
                    if full_tree:
                        ax0.scatter(xcoord_main*np.ones_like(ycoord_main),ycoord_main,
                                    c=sub_tree_main['color_plot'])
                    
                        for k in range(len(ycoord_main)):
                            ax0.plot([xcoord_main,xcoord_main],[ycoord_main[k],yplot],c='grey',zorder=0)
                        
            xcoor_aux=xplot+xcoor_aux2
            
        xplot_main[l]=(xcoor_aux)+50  # El '+50' es para que haya separación suficiente al graficar

    # Scatter plot de la main branch
    ax0.scatter(xplot_main,yplot_main,c=color_plot_main,s=200)
    
    # Plot thicker line for the main branch
    ax0.plot(xplot_main,yplot_main,c='k',lw=3,zorder=0)
    
    # Unir cada progenitor con su descendiente de la main branch
    if full_tree:
        for k in range(len(yplot_main)-1):
            mask=np.where(yplot_prog==yplot_main[k])[0]
            for ind in mask:
                ax0.plot([xplot_prog[ind],xplot_main[k+1]],[yplot_prog[ind],yplot_main[k+1]],
                         c='grey',zorder=0)
    
    # Falso scatter plot para poner color bar
    xplot_false,yplot_false=[],[]
    for k in range(len(color_plot)):
        xplot_false=np.append(xplot_false,None)
        yplot_false=np.append(yplot_false,None)
    sc_bar=ax0.scatter(xplot_false,yplot_false,marker='o',s=0,c=cvar,
                       facecolor=color_plot,vmin=vmin,vmax=vmax,cmap=cmap,)

    # Vertical color bar with reference to redshifts
    cbar=fig0.colorbar(sc_bar, ax=ax0, orientation='vertical',shrink=1,pad=0.01,aspect=30)
    cbar.set_label(color_var_label,fontsize=40,rotation=270,labelpad=50)
    cbar.ax.tick_params(axis='both', which='major', labelsize=30,top=True,bottom=False,
                    labeltop=True,labelbottom=False)  

    # Tune up the plot    
    ax0.set_ylabel(yvar,fontsize=40)
    ymin,ymax=ax0.get_ylim()
    if invert_yaxis:
        ax0.set_ylim(ymax,ymin)
    
    ax0.tick_params(which='major',axis='both',labelsize=30)
    
    plt.subplots_adjust(left=0.05,right=1.05,bottom=0.03,top=0.99)

    ax0.get_xaxis().set_visible(False)
    
    return fig0,ax0

# ================================================================================================================    
# ================================================================================================================   















