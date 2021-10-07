#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:48:02 2021

@author: willy
"""

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import execution_game_automate_4_all_t as autoExeGame4T
import fonctions_auxiliaires as fct_aux

from pathlib import Path

# _________                 merge Cx : debut                       ___________
def resume_stat_N_instances(df, path_2_dir_ROOT, k_steps_max, nb_instances):
    ids = df[df.C4 == k_steps_max-1].index
    df.loc[ids, "C2"] = False
    
    ids = df[~((df.C1 == True) & (df.C2 == True))].index
    df.loc[ids, "C3"] = None
    
    # % de OUI dans C1
    percent_OUI_C1 = round(df[df.C1 == True].C1.count()/nb_instances, 3)
    
    # % de OUI dans C2
    percent_OUI_C2 = round(df[df.C2 == True].C2.count()/nb_instances, 3)
    
    # % de OUI dans C3
    s_C3 = df.C3.dropna()
    percent_OUI_C3 = None
    if s_C3.size == 0:
        percent_OUI_C3 = 0
    else:
        percent_OUI_C3 = round(s_C3[s_C3 == 1.0].count()/s_C3.count(), 3)
        
    
    # C4: moy du nombre d'etapes stabilisées dans C2
    mean_C4 = df[(df.C2 == True)].C4.mean()
    if np.isnan(mean_C4):
        mean_C4 = 0
    
    # moy des Perfs de RF
    mean_perf_C5 = df.C5.mean()
    
    # moy des perfs de BF
    mean_perf_C6 = df.C6.mean()
    
    # moy de ceux ayant un equilibre de Nash cad C1 = True
    mean_Perf_C7 = df[df.C1 == True].C7.mean()
    
    dico = {"percent_OUI_C1": [percent_OUI_C1], 
            "percent_OUI_C2": [percent_OUI_C2],
            "percent_OUI_C3": [percent_OUI_C3],
            "mean_C4": [mean_C4],
            "mean_perf_C5": [mean_perf_C5],
            "mean_perf_C6": [mean_perf_C6], 
            "mean_Perf_C7": [mean_Perf_C7]
            }
    
    df_res = pd.DataFrame(dico).T
    df_res.columns = ["value"]
    
    df_res.to_excel(os.path.join(path_2_dir_ROOT, 
                                 "resume_stat_50_instances.xlsx"))
    
    df.to_excel(os.path.join(path_2_dir_ROOT, 
                             "RESUME_50_INSTANCES.xlsx"))
# _________                 merge Cx : fin                       ___________

    
# _________             define paramters of main : debut            ___________
def define_parameters_MULTI_gammaV_instances_phiname_arrplMTVars(dico_params):
    params = []
    
    for gamma_version in dico_params["gamma_versions"]:
        
        for phi_name, dico_ab in dico_params["dico_phiname_ab"].items():
        
            # ----   execution of 50 instances    ----
            name_dir_oneperiod \
                = os.path.join(
                    dico_params["name_dir"],
                    #"OnePeriod_50instances",
                    phi_name+"OnePeriod_50instances_ksteps"\
                        +str(dico_params["ksteps"])\
                        +"_b"+str(dico_params["learning_rates"][0])\
                        +"_kstoplearn"+str(dico_params["kstoplearn"]),
                    "OnePeriod_"+str(dico_params["nb_instances"])\
                        +"instances"\
                        +"GammaV"+str(gamma_version))
            
            for numero_instance in range(0, dico_params["nb_instances"]):
                arr_pl_M_T_vars_init = None
                if dico_params["doc_VALUES"]==23:
                    arr_pl_M_T_vars_init \
                        = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc23(
                            dico_params["setA_m_players"], 
                            dico_params["setB_m_players"], 
                            dico_params["setC_m_players"], 
                            dico_params["t_periods"], 
                            dico_params["scenario"],
                            dico_params["scenario_name"],
                            dico_params["path_to_arr_pl_M_T"], 
                            dico_params["used_instances"])
                elif dico_params["doc_VALUES"]==24:
                    arr_pl_M_T_vars_init \
                        = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc24(
                            dico_params["setA_m_players"], 
                            dico_params["setB_m_players"], 
                            dico_params["setC_m_players"], 
                            dico_params["t_periods"], 
                            dico_params["scenario"],
                            dico_params["scenario_name"],
                            dico_params["path_to_arr_pl_M_T"], 
                            dico_params["used_instances"])
                elif dico_params["doc_VALUES"]==25:
                    arr_pl_M_T_vars_init \
                        = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc25(
                            dico_params["setA_m_players"], 
                            dico_params["setB_m_players"], 
                            dico_params["setC_m_players"], 
                            dico_params["t_periods"], 
                            dico_params["scenario"],
                            dico_params["scenario_name"],
                            dico_params["path_to_arr_pl_M_T"], 
                            dico_params["used_instances"])
                    
                pi_hp_plus_T, pi_hp_minus_T, \
                phi_hp_plus_T, phi_hp_minus_T \
                    = fct_aux.compute_pi_phi_HP_minus_plus_all_t(
                        arr_pl_M_T_vars_init=arr_pl_M_T_vars_init,
                        t_periods=dico_params["t_periods"],
                        pi_hp_plus=dico_params["pi_hp_plus"][0],
                        pi_hp_minus=dico_params["pi_hp_minus"][0],
                        a=dico_ab['a'],
                        b=dico_ab['b'], 
                        gamma_version=gamma_version, 
                        manual_debug=dico_params["manual_debug"],
                        dbg=dico_params["debug"])
                
                date_hhmm_new = "_".join([date_hhmm, str(numero_instance), 
                                          "t", str(dico_params["t_periods"])])
                
                param = [arr_pl_M_T_vars_init.copy(), 
                         name_dir_oneperiod,
                         date_hhmm_new,
                         k_steps,
                         dico_params["NB_REPEAT_K_MAX"],
                         dico_params["algos"],
                         dico_params["learning_rates"],
                         dico_params["pi_hp_plus"],
                         dico_params["pi_hp_minus"],
                         dico_ab['a'], 
                         dico_ab['b'],
                         pi_hp_plus_T, pi_hp_minus_T,
                         phi_hp_plus_T, phi_hp_minus_T,
                         gamma_version,
                         dico_params["ppi_t_base"],
                         dico_params["used_instances"],
                         dico_params["used_storage_det"],
                         dico_params["manual_debug"], 
                         dico_params["criteria_bf"], 
                         numero_instance,
                         dico_params["debug"] 
                         ]
                
                params.append(param)
                
    return params
# _________             define paramters of main : fin            ___________

# _________               execution procedurale : debut            ___________
def run_procedurale_way(params):
    
    for numero, param in enumerate(params):
        print('11={}, 12={} len_param={}'.format(param[11], param[12], len(param)))
        arr_pl_M_T_vars_init= param[0] 
        name_dir_oneperiod = param[1]
        date_hhmm_new = param[2]
        k_steps = param[3]
        NB_REPEAT_K_MAX = param[4]
        algos_names = param[5]
        learning_rates = param[6]
        pi_hp_plus = param[7]
        pi_hp_minus = param[8]
        a = param[9] 
        b = param[10]
        pi_hp_plus_T = param[11]; pi_hp_minus_T = param[12]
        phi_hp_plus_T = param[13]; phi_hp_minus_T = param[14]
        gamma_version = param[15]; ppi_t_base = param[16]
        used_instances = param[17]
        used_storage_det = param[18]
        manual_debug = param[19]
        criteria_bf = param[20] 
        numero_instance = param[21]
        debug = param[22] 
        autoExeGame4T\
            .execute_BF_NH_LRI_OnePeriod_used_Generated_N_INSTANCES_MULTI(
                arr_pl_M_T_vars_init=param[0],
                name_dir=param[1],
                date_hhmm=param[2],
                k_steps=param[3],
                NB_REPEAT_K_MAX=param[4],
                algos=param[5],
                learning_rates=param[6],
                pi_hp_plus=param[7],
                pi_hp_minus=param[8],
                a=param[9], 
                b=param[10],
                pi_hp_plus_T=param[11], pi_hp_minus_T=param[12],
                phi_hp_plus_T=param[13], phi_hp_minus_T=param[14],
                gamma_version=param[15],
                ppi_t_base=param[16],
                used_instances=param[17],
                used_storage_det=param[18],
                manual_debug=param[19], 
                criteria_bf=param[20], 
                numero_instance=param[21],
                dbg=param[22])
# _________               execution procedurale : fin              ___________

if __name__ == "__main__":
    ti = time.time()
    
    # constances 
    criteria_bf = "Perf_t"
    used_storage_det = True #False #True
    manual_debug = False #True #False #True
    used_instances = False #True
    debug = False
    
    date_hhmm="DDMM_HHMM"
    t_periods = 1 #50 #30 #35 #55 #117 #15 #3
    k_steps = 30000 #250 #50000 #10000 #250 #50000 #250 #5000 #2000 #50 #250 10000
    NB_REPEAT_K_MAX= 3 #10 #3 #15 #30
    learning_rates = [0.01] #[0.01] #[0.01]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    fct_aux.N_DECIMALS = 8
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}, "A1.2B0.8": {"a":1.2, "b":0.8}}
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}}
    pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    pi_hp_minus = [30] #[20] #[0.33] #[15, 5]
    fct_aux.PI_0_PLUS_INIT = 4 #20 #4
    fct_aux.PI_0_MINUS_INIT = 3 #10 #3
    doc_VALUES = 25 #24

    NB_INSTANCES = 10 #50
            
    
    algo_names = fct_aux.ALGO_NAMES_LRIx + fct_aux.ALGO_NAMES_DET \
                + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
                
    algo_names = [fct_aux.ALGO_NAMES_LRIx[1]] \
                    + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
            
    
            
    # ---- initialization of variables for generating instances ----
    setA_m_players, setB_m_players, setC_m_players = 15, 10, 10                # 35 players 
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5                  # 21 players 
    setA_m_players, setB_m_players, setC_m_players = 8, 4, 4                   # 16 players
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
    setA_m_players, setB_m_players, setC_m_players = 4, 3, 3                   # 10 players
    #setA_m_players, setB_m_players, setC_m_players = 2, 2, 2                   # 6 players
    #setA_m_players, setB_m_players, setC_m_players = 1, 1, 1                   # 3 players
                      
    scenario_name = "scenarioOnePeriod"
    scenario = None
    
    name_dir = "tests"
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    gamma_versions = [-2] #-1 : random normal distribution, 0: not stock anticipation, -2: normal distribution with proba ppi_k
    ppi_t_base = 0.3 # None par defaut
    
    dico_params = {"dico_phiname_ab":dico_phiname_ab, "doc_VALUES":doc_VALUES,
        "gamma_versions":gamma_versions,
        "nb_instances":NB_INSTANCES,
        "criteria_bf":criteria_bf, "used_storage_det":used_storage_det,
        "manual_debug":manual_debug, "debug":debug,
        "date_hhmm":date_hhmm,
        "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus,
        "algos":algo_names, "learning_rates":learning_rates,
        "setA_m_players":setA_m_players, 
        "setB_m_players":setB_m_players, 
        "setC_m_players":setC_m_players,
        "t_periods":t_periods,"scenario":scenario,"scenario_name":scenario_name,
        "path_to_arr_pl_M_T":path_to_arr_pl_M_T,"used_instances":used_instances, 
        "name_dir":name_dir, 
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX, 
        "ksteps": k_steps, "learning_rates":learning_rates, 
        "kstoplearn": fct_aux.STOP_LEARNING_PROBA, 
        "ppi_t_base": ppi_t_base}
    
    params = define_parameters_MULTI_gammaV_instances_phiname_arrplMTVars(dico_params)
    print("define parameters finished")
    
    # run_procedurale_way(params)
    
    # multi processing execution : debut
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(
        autoExeGame4T.execute_BF_NH_LRI_OnePeriod_used_Generated_N_INSTANCES_MULTI,
        params
    )
    # multi processing execution : fin
    
    # merge all Cx
    for gamma_version, learning_rate in zip(gamma_versions, learning_rates):
        for phi_name_ab, dico_ab in dico_phiname_ab.items():
            #phi_name = "A1B1"
            name_rep = phi_name_ab+"OnePeriod_50instances_ksteps"\
                                +str(k_steps)\
                                +"_b"+str(learning_rate)\
                                +"_kstoplearn"+str(fct_aux.STOP_LEARNING_PROBA)
                                
            path_2_dir_ROOT = os.path.join("tests", name_rep)
            file_2_save_all_instances = "save_all_instances"
            path_2_dir = os.path.join(path_2_dir_ROOT,
                                      file_2_save_all_instances)
            files_csv = os.listdir(path_2_dir)
            
            name_cols = [fct_aux.name_cols_CX["C1"], fct_aux.name_cols_CX["C2"], 
                          fct_aux.name_cols_CX["C3"], fct_aux.name_cols_CX["C4"], 
                          fct_aux.name_cols_CX["C5"], fct_aux.name_cols_CX["C6"], 
                          fct_aux.name_cols_CX["C7"], 
                          fct_aux.name_cols_CX["C9"], 
                          fct_aux.name_cols_CX["check_C5_inf_C6"], 
                          fct_aux.name_cols_CX["check_C7_inf_C6"] ] 
            df = pd.DataFrame(columns=name_cols)
            for file_csv in files_csv:
                df_tmp = pd.read_csv(os.path.join(path_2_dir, file_csv), index_col=0)
                df = pd.concat([df, df_tmp])
            
            resume_stat_N_instances(df=df, path_2_dir_ROOT=path_2_dir_ROOT, 
                                    k_steps_max=k_steps, nb_instances=NB_INSTANCES)
    print("Multi process running time ={}".format(time.time()-ti))
    
    