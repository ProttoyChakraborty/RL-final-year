from envSim import EnvData
import pandas as pd 

env= EnvData("MountainCar-v0")
env.simulate(10)
data=env.getData()
data.to_csv("Data.csv")
env.describe_state_space()