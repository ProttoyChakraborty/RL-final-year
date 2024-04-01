import os
import pandas as pd
import numpy as np


class Env:
    def __init__(self,envDataFile):
        """Create an Env from an EnvData csv file

        Parameters
        ----------
        envData(str) : Path of the file that contains the envData dataframe in csv format
        """
        if(not envDataFile):
            raise ValueError("cannot create Env object as no EnvData file is provided")

        self.data=pd.read_csv(envDataFile)
        self.no_of_states=self.data.filter(regex=("^S.*")).count(axis=1)
        self.state_map={}
        self.bins_map={}
        self.mat =[]

    def discretize_column(column,options,no_of_bins=0 ,width_of_bins=0,bin_boundaries=[]):
        """ discretize a column using it's name and options for discretizing 

        Parameters
        ----------

        colname(str) : Name of the column to be discretized.

        option(str) : type of discretization , it can be assigned to 

                * "fixed_number" : Binning into a fixed number of bins.
                * "fixed_width" : Binning into bins witt fixed width.
                * "flexible" : Binning into bins with flexible bin boundaries.
        
        no_of_bins(int) : if option is "fixed_number", input the number of required bins.
        
        width_of_bins(float64) : if option is "fixed_width", input the width of a bin.
        
        bin_boundaries(Array<Tuple(float64,float64)>) : if option is "flexible", input 
                                                    the bin boundaries    

        Returns
        -------
        
        >>> Discretized Series
        """
        if(options=="fixed_number"):
            if(not no_of_bins or no_of_bins<1):
                raise AttributeError("Invalid no of bins argument : no_of_bins must be defined and greater than 1 ")
            column,bins=pd.cut(column,bins=no_of_bins,labels=[x+1 for x in range(no_)],retbins=True)
            self.bins=bins
            return column
        elif (options=="fixed_width"):
            print(width_of_bins)
            if(not width_of_bins or width_of_bins<=0):
                raise AttributeError("Invalid width argument : width_of_bins must be defined and be greater than 0")
            n=int((column.max()-column.min())/width_of_bins)
            column,bins=pd.cut(column,bins=n,labels=[x+1 for x in range(n)],retbins=True)
            self.bins=bins
            return column
        elif(options=="flexible"):
            if(not bin_boundaries or len(bin_boundaries)<=1):
                raise ValueError("Invalid Argument for bin boundaries")
            column,bins=pd.cut(column,bins=pd.IntervalIndex.from_tuples(bin_boundaries),labels=[x+1 for x in range(len(bin_boundaries))],retbins=True)
            self.bins=bins
            return column
        else:
            raise ValueError("Invalid value for 'options' parameter")
        

    def descretize_state_variable(self,col,option,no_of_bins=0 ,width_of_bins=0,bin_boundaries=[]):
        """
        Transform a particular state variable into discrete series in both inital and final states
        
        Parameter
        ---------
        self.data( Pandas.Dataframe ) : The dataframe to apply the transformation

        cols (List<int>) : List of index of state variables to be transformed , e.g if S0,S1,S2 need to be 
                        transformed use [0,1,2] as input. Hence, S0,S1,S2 and T0,T1,T2 will be transformed.  

        option(str) : type of discretization , it can be assigned to 

                    * "fixed_number" : Binning into a fixed number of bins.
                    * "fixed_width" : Binning into bins witt fixed width.
                    * "flexible" : Binning into bins with flexible bin boundaries.
            
        no_of_bins(int) : if option is "fixed_number", input the number of required bins.
            
        width_of_bins(float64) : if option is "fixed_width", input the width of a bin.
            
        bin_boundaries(Array<Tuple(float64,float64)>) : if option is "flexible", input 
                                                        the bin boundaries    

        Returns
        -------
            
        Dataframe with discretised columns
        """

        
        intial_state_colname="S{}".format(col)
        final_state_colname="T{}".format(col)
        self.data[intial_state_colname+'_d']=self.discretize_column(self.data[intial_state_colname], options,no_of_bins,width_of_bins,bin_boundaries)
        self.data[final_state_colname+'_d']=self.discretize_column(self.data[final_state_colname], options,no_of_bins,width_of_bins,bin_boundaries)
        
    def discretize_state_space(self,options):
        """
        Discretizes the entire state space based on options provided
        each state variable must have an option object associated with it.
        Parameter
        ---------
        options (JSON): [{
                            state(str) : State to be transformed i.e "S1","S2"...
                                      

                            option(str) : type of discretization , it can be assigned to 

                                        * "fixed_number" : Binning into a fixed number of bins.
                                        * "fixed_width" : Binning into bins witt fixed width.
                                        * "flexible" : Binning into bins with flexible bin boundaries.
                                
                            no_of_bins(int) : if option is "fixed_number", input the number of required bins.
                                
                            width_of_bins(float64) : if option is "fixed_width", input the width of a bin.
                                
                            bin_boundaries(Array<Tuple(float64,float64)>) : if option is "flexible", input 
                                                                            the bin boundaries
                        }]

        """
        if len(options)!=self.no_of_states:
            raise AttributeError(f"{len(options)} options provided, {self.no_of_states} required")
        else:
            for option in options:
                state_n=int(option["state"][-1])
                type_arg=option.type
                n_bins=0 if option.no_of_bins is None else option.no_of_bins
                w_bins=0 if option.width_of_bins is None else option.width_of_bins
                bin_bounds=[] if option.bin_boundaries is None else option.bin_boundaries
                try:
                    self.descretize_state_variable(col=[state_n],option=type_arg,no_of_bins=n_bins,width_of_bins=w_bins,bin_boundaries=bin_bounds)
                except Error as e:
                    print("An error occurred while processing {}".format(state_n))
                    return
            print("Succesfully processed states!")
                
            

    def transform_state_variables_to_state(self):
        self.data["I_STATE"]=self.data[self.data.filter(regex=("^S.*"))].apply(lambda x: "-".join(x), axis =1)
        self.data["F_STATE"]=self.data[self.data.filter(regex=("^T.*"))].apply(lambda x: "-".join(x), axis =1)

    def make_states_from_discrete(self,cols):
        i_states=["S{}_d".format(x) for x in cols]
        f_states=["T{}_d".format(x) for x in cols]
        self.data["I_STATE"]=self.data[i_states].astype(str).apply(lambda x:"_".join(x),axis=1)
        self.data["F_STATE"]=self.data[f_states].astype(str).apply(lambda x:"_".join(x),axis=1)
        unique_states=pd.unique(pd.concat([self.data["I_STATE"],self.data["F_STATE"]],axis=0))
        for i,s in enumerate(unique_states):
            self.state_map[s]=i
        self.data["I_STATE_M"]=self.data["I_STATE"].map(lambda x:self.state_map[x])
        self.data["F_STATE_M"]=self.data["F_STATE"].map(lambda x:self.state_map[x])
    
    def get_MDP(self):
        i_map=self.data["I_STATE_M"].value_counts()
        graph={}
        for i in range(len(self.data)):
            i_state=self.data["I_STATE_M"][i]
            f_state=self.data["F_STATE_M"][i]
            if((i_state,f_state) not in graph.keys()):
                t=i_map[i_state]
                s1=f"I_STATE_M=={i_state}"
                s2=f"F_STATE_M=={f_state}"
                qs=s1+"&"+s2
                res=len(self.data.query(qs))
                graph[(i_state,f_state)]=round(res/t,2)
