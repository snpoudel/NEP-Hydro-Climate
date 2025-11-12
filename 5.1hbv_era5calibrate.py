#import packages
import numpy as np
import pandas as pd
import multiprocessing as mp
from hbv_model import hbv #imported from local python script
from geneticalgorithm import geneticalgorithm as ga # install package first
import warnings
warnings.filterwarnings("ignore")

#function that reads station ID, takes input for that ID and outputs calibrated parameters and nse values
def calibNSE(station_id, start_train_year, end_train_year):
    #read input csv file
    df = pd.read_csv(f"data/processed_data/era5/input{station_id}.csv")
    #only used data upto 2005 for calibration
    df = df[(df['date'] >= f'{start_train_year}') & (df['date'] <= f'{end_train_year}')].reset_index(drop=True)
    #hbv model input
    p = df["precip"]
    temp = df["temp"]
    date = df["date"]
    latitude = df["latitude"]
    routing = 1 # 0: no routing, 1 allows running
    q_obs = df["Streamflow"] #validation data / observed flow

    ##genetic algorithm for hbv model calibration
    #reference: https://github.com/rmsolgi/geneticalgorithm
    #write a function you want to minimize
    def nse(pars):
        q_sim = hbv(pars, p, temp, date, latitude, routing)
        #dont use first year data for calibration, spinup period
        q_sim = q_sim[365:]
        q_obs_inner = q_obs[365:]
        denominator = np.sum((q_obs_inner - (np.mean(q_obs_inner)))**2)
        numerator = np.sum((q_obs_inner - q_sim)**2)
        nse_value = 1 - (numerator/denominator)
        return -nse_value #minimize this (use negative sign if you need to maximize)

    varbound = np.array([[1,1000], #fc
                        [1,7], #beta
                        [0.01,0.99], #pwp
                        [1,999], #l
                        [0.01,0.99], #ks
                        [0.01,0.99], #ki
                        [0.0001, 0.99], #kb
                        [0.001, 0.99], #kperc
                        [0.5,2], #coeff_pet
                        [0.01,10], #ddf
                        [0.5,1.5], #scf
                        [-1,4], #ts
                        [-1,4], #tm
                        [0,4], #tti
                        [0, 0.2], #whc
                        [0.1,1], #crf
                        [1,10]]) #maxbas

    algorithm_param = {
        'max_num_iteration': 50,#100              # Generations, higher is better, but requires more computational time
        'max_iteration_without_improv': None,   # Stopping criterion for lack of improvement
        'population_size': 1000,#500                 # Number of parameter-sets in a single iteration/generation(to start with population 10 times the number of parameters should be fine!)
        'parents_portion': 0.3,                 # Portion of new generation population filled by previous population
        'elit_ratio': 0.01,                     # Portion of the best individuals preserved unchanged
        'crossover_probability': 0.3,           # Chance of existing solution passing its characteristics to new trial solution
        'crossover_type': 'uniform',            # Create offspring by combining the parameters of selected parents
        'mutation_probability': 0.01            # Introduce random changes to the offspring’s parameters (0.1 is 10%)
    }

    model = ga(function = nse,
            dimension = 17, #number of parameters to be calibrated
            variable_type= 'real',
            variable_boundaries = varbound,
            algorithm_parameters = algorithm_param)

    model.run()
    #end of genetic algorithm

    #output of the genetic algorithm/best parameters
    best_parameters = model.output_dict
    param_value = best_parameters["variable"]
    nse_value = best_parameters["function"]
    nse_value = -nse_value #nse function gives -ve values, which is now reversed here to get true nse
    #convert into a dataframe
    df_param = pd.DataFrame(param_value).transpose()
    df_param = df_param.rename(columns={0:"fc", 1:"beta", 2:"pwp", 3:"l", 4:"ks", 5:"ki",
                             6:"kb", 7:"kperc",  8:"coeff_pet", 9:"ddf", 10:"scf", 11:"ts",
                             12:"tm", 13:"tti", 14:"whc", 15:"crf", 16:"maxbas"})
    df_param["station_id"] = str(station_id)
    df_nse = pd.DataFrame([nse_value], columns=["nse"])
    df_nse["station_id"] = str(station_id)
    #save as a csv file
    df_param.to_csv(f"output/parameter/param_{station_id}.csv", index = False)
    df_nse.to_csv(f"output/nse/nse_{station_id}.csv", index = False)
    #End of function


def run_calibration(station_id, train_date):
    #unpack the tuple
    start_train_year, end_train_year = train_date
    calibNSE(station_id, start_train_year, end_train_year)
    print(f"Calibration for station {station_id} is done!")

if __name__ == "__main__":
    #read basin list
    basin_list = pd.read_csv('data/basins_og.csv')
    #only keep mardi and chepe basins
    # basin_list = basin_list[basin_list['name'].isin(['mardi', 'chepe'])]
    pool = mp.Pool(mp.cpu_count())

    tasks = [(station_id, basin_list[basin_list['id'] == station_id][['start_train_date', 'end_train_date']].values[0]) for station_id in basin_list['id']]
    pool.starmap(run_calibration, tasks)

    pool.close()
    pool.join()