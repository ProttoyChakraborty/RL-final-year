import numpy as np 
import pandas as pd

def discretize_column(column,options,noOfBins=0 ,widthOfBins=0,binBoundaries=[]):
        """ discretize a column using it's name and options for discretizing 

        Parameters
        ----------

        colname(str) : Name of the column to be discretized.

        option(str) : type of discretization , it can be assigned to 

                * "fixed_number" : Binning into a fixed number of bins.
                * "fixed_width" : Binning into bins witt fixed width.
                * "flexible" : Binning into bins with flexible bin boundaries.
        
        noOfBins(int) : if option is "fixed_number", input the number of required bins.
        
        widthOfBins(float64) : if option is "fixed_width", input the width of a bin.
        
        binBoundaries(Array<Tuple(float64,float64)>) : if option is "flexible", input 
                                                    the bin boundaries    

        Returns
        -------
        
        >>> Discretized Series
        """
        if(options=="fixed_number"):
            if(not noOfBins or noOfBins<1):
                raise AttributeError("Invalid no of bins argument : noOfBins must be defined and greater than 1 ")
            column,bins=pd.cut(column,bins=noOfBins,retbins=True,labels=[x+1 for x in range(noOfBins)])
            return column,bins
        elif (options=="fixed_width"):
            print(widthOfBins)
            if(not widthOfBins or widthOfBins<=0):
                raise AttributeError("Invalid width argument : widthOfBins must be defined and be greater than 0")
            n=int((column.max()-column.min())/widthOfBins)
            column,bins=pd.cut(column,retbins=True,bins=n,labels=[x+1 for x in range(n)])
            return column,bins
        elif(options=="flexible"):
            if(not binBoundaries or len(binBoundaries)<=1):
                raise ValueError("Invalid Argument for bin boundaries")
            column,bins=pd.cut(column,retbins=True,bins=pd.IntervalIndex.from_tuples(binBoundaries),labels=[x+1 for x in range(len(binBoundaries))])
            return column,bins
        else:
            raise ValueError("Invalid value for 'options' parameter")
        


def descretize_state_variable(df,cols,options,noOfBins=0 ,widthOfBins=0,binBoundaries=[]):
    """
    Transform a particular state variable into discrete series in both inital and final states
    
    Parameter
    ---------
    df( Pandas.Dataframe ) : The dataframe to apply the transformation

    cols (List<int>) : List of index of state variables to be transformed , e.g if S0,S1,S2 need to be 
                    transformed use [0,1,2] as input. Hence, S0,S1,S2 and T0,T1,T2 will be transformed.  

    option(str) : type of discretization , it can be assigned to 

                * "fixed_number" : Binning into a fixed number of bins.
                * "fixed_width" : Binning into bins witt fixed width.
                * "flexible" : Binning into bins with flexible bin boundaries.
        
    noOfBins(int) : if option is "fixed_number", input the number of required bins.
        
    widthOfBins(float64) : if option is "fixed_width", input the width of a bin.
        
    binBoundaries(Array<Tuple(float64,float64)>) : if option is "flexible", input 
                                                    the bin boundaries    

    Returns
    -------
        
    >>> Dataframe with discretised columns
    """

    for x in cols :
        intial_state_colname="S{}".format(x)
        final_state_colname="T{}".format(x)
        df[intial_state_colname+"_d"],bins=discretize_column(df[intial_state_colname], options,noOfBins,widthOfBins,binBoundaries)
        df[final_state_colname+"_d"],bins=discretize_column(df[final_state_colname], options,noOfBins,widthOfBins,binBoundaries)
    
    return df,bins



def make_states_from_discrete(df,cols):
    state_map={}
    i_states=["S{}_d".format(x) for x in cols]
    f_states=["T{}_d".format(x) for x in cols]
    df["I_STATE"]=df[i_states].astype(str).apply(lambda x:"_".join(x),axis=1)
    df["F_STATE"]=df[f_states].astype(str).apply(lambda x:"_".join(x),axis=1)
    unique_states=pd.unique(pd.concat([df["I_STATE"],df["F_STATE"]],axis=0))
    for i,s in enumerate(unique_states):
        state_map[s]=i
    df["I_STATE_M"]=df["I_STATE"].map(lambda x:state_map[x])
    df["F_STATE_M"]=df["F_STATE"].map(lambda x:state_map[x])
    return df,state_map


def get_MDP(df):
    n=df["I_STATE"].nunique()
    mat = np.zeros([n,n],dtype=int)
    for i in range(n):
        t=len(df.loc(df["I_STATE"]==i))
        for j in range(n):
            mat[i][j]=len(df.loc((df["I_STATE"]==i)&(df["F_STATE"]==j)))/t
    return mat
    
