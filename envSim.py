import os
import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime
import traceback


class EnvData:
    def __init__(self, name: str):
        self.columns = []
        self.rows = []
        self.envName = name
        self.env = gym.make(self.envName)
        self.actions = []
        self.episodes = 0
        self.data = None
        self.mdp = {}
        self.no_of_states = 0
        self.state_map = {}
        self.bins_map = {}
        self.bin_values = {}
        self.mat = []
        self.states_to_use = []
        self.init()

    def init(self):
        obs = self.env.reset()[0]
        print(len(obs))
        self.no_of_states = len(obs)
        self.columns.append("Episode")
        self.columns.append("TimeStep")
        for i in range(len(obs)):
            self.columns.append("S{}".format(i))
        self.columns.append("Action")
        self.columns.append("Reward")
        for i in range(len(obs)):
            self.columns.append("T{}".format(i))

    def addRow(self, episode, time, curr_state, action, reward, observation):
        row = []
        row.append(episode)
        row.append(time)
        for x in curr_state:
            row.append(x)
        if action not in self.actions:
            self.actions.append(action)
        row.append(action)
        row.append(reward)
        for x in observation:
            row.append(x)
        if (len(row) != len(self.columns)):
            raise Exception("row and column lengths do not match")
        self.rows.append(row)

    def simulate(self, episodes: int):
        for i in range(episodes):
            time = 0
            curr_state = self.env.reset()[0]
            while (True):
                time += 1
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(
                    action)
                self.addRow(i, time, curr_state, action, reward, observation)
                curr_state = observation
                if terminated or truncated:
                    break

    def getData(self):
        self.data = pd.DataFrame(self.rows, columns=self.columns)
        return self.data

    def save_data_to_file(self):
        filename = "{}_simulation_{}.csv".format(
            self.envName, datetime.now().strftime("%Y%m%d-%H%M"))
        self.getData().to_csv(filename)
        return filename

    def describe_state_space(self):
        df = self.getData()
        df = df.filter(regex=("^{S,T}*"))
        print(df.describe())

    def i_states(self):
        colnames = [f"S{x}_d" for x in self.states_to_use]
        res = []
        for i in range(len(self.data)):
            states = []
            for col in colnames:
                state = self.data[col].iloc[i]
                states.append(self.bin_values[col[:2]][int(state)-1])
                # col[:2] fetches the first 2 chars from colname i.e S0 from S0_d
            res.append(states)
        self.data[f"I_STATES_APPROX"] = pd.Series(res)

    def f_states(self):
        colnames = [f"T{x}_d" for x in self.states_to_use]
        print(colnames)
        res = []
        for i in range(len(self.data)):
            states = []
            for col in colnames:
                state = self.data[col].iloc[i]
                states.append(self.bin_values[col[:2]][int(state)-1])
                # col[:2] fetches the first 2 chars from colname i.e T0 from T0_d
            res.append(states)
        self.data[f"F_STATES_APPROX"] = pd.Series(res)

    def make_state_vectors(self):
        self.i_states()
        self.f_states()
        print("created approximate state vectors...")

    def discretize_column(self, column_name, column, options, no_of_bins=0, width_of_bins=0, bin_boundaries=[]):
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
        is_final = True if "T" in column_name else False
        if (options == "fixed_number"):
            bins = no_of_bins
            if (not no_of_bins or (isinstance(no_of_bins, int) and no_of_bins < 1)):
                raise AttributeError(
                    "Invalid no of bins argument : no_of_bins must be defined and greater than 1 ")
            if is_final:
                bins = self.bins_map[f"S{column_name[-1]}"]
            column, bins = pd.cut(column, bins=bins, labels=[
                                  x+1 for x in range(no_of_bins)], retbins=True)
            self.bins_map[column_name] = bins.tolist()
            self.bin_values[column_name] = self.process_bins(bins.tolist())
            return column
        elif (options == "fixed_width"):
            if (not width_of_bins or width_of_bins <= 0):
                raise AttributeError(
                    "Invalid width argument : width_of_bins must be defined and be greater than 0")
            n = int((column.max()-column.min())/width_of_bins)
            bins = n
            if is_final:
                bins = self.bins_map[f"S{column_name[-1]}"]
            column, bins = pd.cut(column, bins=bins, labels=[
                                  x+1 for x in range(n)], retbins=True)
            self.bins_map[column_name] = bins.tolist()
            self.bin_values[column_name] = self.process_bins(bins.tolist())
            return column
        elif (options == "flexible"):
            if (not bin_boundaries or len(bin_boundaries) <= 1):
                raise ValueError("Invalid Argument for bin boundaries")
            column, bins = pd.cut(column, bins=pd.IntervalIndex.from_tuples(
                bin_boundaries), labels=[x+1 for x in range(len(bin_boundaries))], retbins=True)
            self.bins_map[column_name] = bins.tolist()
            self.bin_values[column_name] = self.process_bins(bins.tolist())
            return column
        else:
            raise ValueError("Invalid value for 'options' parameter")

    def descretize_state_variable(self, col, option, no_of_bins=0, width_of_bins=0, bin_boundaries=[]):
        """
        Transform a particular state variable into discrete series in both inital and final states

        Parameter
        ---------
        self.data( Pandas.Dataframe ) : The dataframe to apply the transformation

        col (int) : Index of state variables to be transformed , e.g if S0 need to be 
                        transformed use 0 as input. Hence, S0 and T0 will be transformed.  

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

        intial_state_colname = "S{}".format(col)
        final_state_colname = "T{}".format(col)
        self.data[intial_state_colname+'_d'] = self.discretize_column(
            intial_state_colname, self.data[intial_state_colname], option, no_of_bins, width_of_bins, bin_boundaries)
        self.data[final_state_colname+'_d'] = self.discretize_column(
            final_state_colname, self.data[final_state_colname], option, no_of_bins, width_of_bins, bin_boundaries)
        self.data.dropna(inplace=True)

    def discretize_state_space(self, options):
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
        for option in options:
            state_n = int(option["state"][-1])
            self.states_to_use.append(state_n)
            type_arg = option["option"]
            n_bins = 0 if "no_of_bins" not in option.keys(
            ) else option["no_of_bins"]
            w_bins = 0 if "width_of_bins" not in option.keys(
            ) else option["width_of_bins"]
            bin_bounds = [] if "bin_boundaries" not in option.keys() else option["bin_boundaries"]
            try:
                self.descretize_state_variable(
                    col=state_n, option=type_arg, no_of_bins=n_bins, width_of_bins=w_bins, bin_boundaries=bin_bounds)
            except Exception as e:
                print("An error occurred while processing {}".format(state_n))
                print(traceback.format_exception(e))
                return
        print("Succesfully processed states!")

    def transform_state_variables_to_state(self):
        self.data["I_STATE"] = self.data[self.data.filter(
            regex=("^S.*"))].apply(lambda x: "-".join(x), axis=1)
        self.data["F_STATE"] = self.data[self.data.filter(
            regex=("^T.*"))].apply(lambda x: "-".join(x), axis=1)

    def make_states_from_discrete(self):
        i_states = ["S{}_d".format(x) for x in self.states_to_use]
        f_states = ["T{}_d".format(x) for x in self.states_to_use]
        self.data["I_STATE"] = self.data[i_states].astype(
            str).apply(lambda x: "_".join(x), axis=1)
        self.data["F_STATE"] = self.data[f_states].astype(
            str).apply(lambda x: "_".join(x), axis=1)
        unique_states = pd.unique(
            pd.concat([self.data["I_STATE"], self.data["F_STATE"]], axis=0))
        for i, s in enumerate(unique_states):
            q1 = self.data.loc[self.data["I_STATE"] == s]
            q2 = self.data.loc[self.data["F_STATE"] == s]
            initial_vec = []
            if (len(q1) > 0):
                initial_vec = q1["I_STATES_APPROX"].tolist()
            else:
                initial_vec = q2["F_STATES_APPROX"].tolist()

            self.state_map[s] = {
                "id": i,
                "state": initial_vec[0],
            }
        self.data["I_STATE_M"] = self.data["I_STATE"].map(
            lambda x: self.state_map[x]["id"])
        self.data["F_STATE_M"] = self.data["F_STATE"].map(
            lambda x: self.state_map[x]["id"])

    def get_MDP(self):
        # get all possible combinations of State,Action,Final
        # if s1 has probabilty p1 to go to f1 on taking action a1 and p2 to go to f2 on taking action a2
        graph = {}
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            i_state = row["I_STATE_M"]
            f_state = row["F_STATE_M"]
            k = row["Action"]
            r = row["Reward"]
            if ((i_state, k, f_state) not in graph.keys()):
                q = self.data.query(f"I_STATE_M=={i_state}&Action=={k}")
                res = q.query(f"F_STATE_M=={f_state}")
                graph[(i_state, k, f_state)] = {
                    "p": round(len(res)/len(q), 2), "reward": r}
        self.mdp = graph

    def process_bins(self, arr):
        """convert an array representing bins to an array representing central observation in each bin"""
        res = []
        for i in range(len(arr)-1):
            res.append((arr[i]+arr[i+1])/2)
        return res
