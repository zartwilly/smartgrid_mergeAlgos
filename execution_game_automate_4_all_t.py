#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:19:26 2021

@author: willy
"""
import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp

import fonctions_auxiliaires as fct_aux
import deterministic_game_model_automate_4_all_t as autoDetGameModel
import lri_game_model_1t as onePeriodLriGameModel
#import force_brute_game_model_automate_4_all_t as autoBfGameModel
#import detection_nash_game_model_automate_4_all_t as autoNashGameModel

import bf_nash_game_modele_1t as bfNhGameModel

from datetime import datetime
from pathlib import Path

def look4BF_NH(algos):
    """
    look for names of Bruce force and nash algorithms
    """
    algos_BF_NH = []
    for algo in algos:
        if algo in fct_aux.ALGO_NAMES_BF or algo in fct_aux.ALGO_NAMES_NASH:
            algos_BF_NH.append(algo)
            
    return list(set(algos_BF_NH))

#_______       all BF, NH, LRI N instances MULTI_PROCESS : debut         ______ 
def execute_BF_NH_LRI_OnePeriod_used_Generated_N_INSTANCES_MULTI(arr_pl_M_T_vars_init,
                                            name_dir=None,
                                            date_hhmm=None,
                                            k_steps=None,
                                            NB_REPEAT_K_MAX=None,
                                            algos=None,
                                            learning_rates=None,
                                            pi_hp_plus=None,
                                            pi_hp_minus=None,
                                            a=1, b=1,
                                            pi_hp_plus_T=None, pi_hp_minus_T=None,
                                            phi_hp_plus_T=None, phi_hp_minus_T=None,
                                            gamma_version=1,
                                            used_instances=True,
                                            used_storage_det=True,
                                            manual_debug=False, 
                                            criteria_bf="Perf_t", 
                                            numero_instance=0,
                                            dbg=False):
    """
    execute algos by using generated instances if there exists or 
        by generating new instances
    
    date_hhmm="1041"
    algos=["LRI1"]
    
    """
    # One Period
    t = 0
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # directory to save  execution algos
    name_dir = "tests" if name_dir is None else name_dir
    date_hhmm = datetime.now().strftime("%d%m_%H%M") \
            if date_hhmm is None \
            else date_hhmm
    
    # steps of learning
    k_steps = 5 if k_steps is None else k_steps
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    # list of algos
    ALGOS = fct_aux.ALGO_NAMES_LRIx + fct_aux.ALGO_NAMES_DET \
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [0.2*pow(10,-3)] if pi_hp_plus is None else pi_hp_plus
    pi_hp_minus = [0.33] if pi_hp_minus is None else pi_hp_minus
    # learning rate 
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates # list(np.arange(0.05, 0.15, step=0.05))
    
    
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    piHpPlusMinus_learningRate = it.product(zip_pi_hp, learning_rates)
    
    # Cx = dict(); 
    # C1 = None; C2 = None; C3 = None; C4 = None; C5 = None; C6 = None;  C7 = None;
    # profils_stabilisation_LRI2 = None; profils_NH = None; 
    # k_stop_learning_LRI2 = None; Perf_sum_Vi_LRI2 = None;
    # Perf_best_profils_bf = None; nb_best_profils_bf = None;
    # Perf_bad_profils_bf = None; nb_bad_profils_bf = None;
    # Perf_bad_profils_NH = None; nb_bad_profils_NH = None;
    
    for ( (pi_hp_plus_elt, pi_hp_minus_elt), learning_rate) \
        in piHpPlusMinus_learningRate:
        
        print("______ execution {}, rate={}______".format(cpt, learning_rate))
        cpt += 1
        
        Cx = dict(); 
        C1 = None; C2 = None; C3 = None; C4 = None; C5 = None; C6 = None;  C7 = None;
        profils_stabilisation_LRI2 = None; profils_NH = None; 
        k_stop_learning_LRI2 = None; Perf_sum_Vi_LRI2 = None;
        Perf_best_profils_bf = None; nb_best_profils_bf = None;
        Perf_bad_profils_bf = None; nb_bad_profils_bf = None;
        Perf_bad_profils_NH = None; nb_bad_profils_NH = None;
        
        pi_0_plus_t = fct_aux.PI_0_PLUS_INIT #4
        pi_0_minus_t = fct_aux.PI_0_MINUS_INIT #3
        pi_hp_plus_t = pi_hp_plus_T[t]
        pi_hp_minus_t = pi_hp_minus_T[t]
    
        arr_pl_MTvars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_vars_init.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus_t=pi_hp_plus_t, 
                                pi_hp_minus_t=pi_hp_minus_t,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)               
        
        # ____      Brute force and Nash equilibrium: debut             _____
        # dico of wrapping up execution for all algos
        dico_profils_BF = {"nb_profils":0, "profils":[], "Perfs":[]}
        dico_profils_NH = {"nb_profils":0, "profils":[], "Perfs":[]}
        dico_best_profils_BF = {"nb_best_profils":0, "profils":[], 
                                "Perfs":[],"nashs":[], "Perfs_nash":[]}
        dico_bad_profils_BF = {"nb_bad_profils":0,"profils":[], 
                                "Perfs":[],"nashs":[], "Perfs_nash":[]}
        dico_mid_profils_BF = {"nb_mid_profils":0,"profils":[], 
                                "Perfs":[],"nashs":[], "Perfs_nash":[]}
        dico_best_profils_NH = {"nb_best_profils":0, "profils":[], "Perfs":[]}
        dico_bad_profils_NH = {"nb_bad_profils":0, "profils":[], "Perfs":[]}
        dico_mid_profils_NH = {"nb_mid_profils":0, "profils":[], "Perfs":[]}
            
        # balanced player game at instant t
        list_dico_modes_profs_by_players_t_bestBF = list()
        list_dico_modes_profs_by_players_t_badBF = list()
        list_dico_modes_profs_by_players_t_midBF = list()
        list_dico_modes_profs_by_players_t_bestNH = list()
        list_dico_modes_profs_by_players_t_badNH = list()
        list_dico_modes_profs_by_players_t_midNH = list()
        
        list_dico_modes_profs_by_players_t_bestBF, \
        list_dico_modes_profs_by_players_t_badBF, \
        list_dico_modes_profs_by_players_t_midBF, \
        list_dico_modes_profs_by_players_t_bestNH, \
        list_dico_modes_profs_by_players_t_badNH, \
        list_dico_modes_profs_by_players_t_midNH, \
        keys_best_BF_NH, keys_bad_BF_NH \
                = bfNhGameModel.generer_balanced_players_4_modes_profils(
                    arr_pl_MTvars_modif, 
                    m_players, t,
                    pi_hp_plus_elt, pi_hp_minus_elt,
                    a, b,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
                
        dico_profils_BF["nb_profils"] = len(list_dico_modes_profs_by_players_t_bestBF) \
                                        + len(list_dico_modes_profs_by_players_t_badBF) \
                                        + len(list_dico_modes_profs_by_players_t_midBF)
        keys_best_NH = set(map(lambda x:x[0], list_dico_modes_profs_by_players_t_bestNH))
        keys_bad_NH = set(map(lambda x:x[0], list_dico_modes_profs_by_players_t_badNH))
        keys_mid_NH = set(map(lambda x:x[0], list_dico_modes_profs_by_players_t_midNH))
        dico_profils_NH["nb_profils"] = len(keys_best_NH.union( keys_bad_NH.union(keys_mid_NH) ))
                       
        # ____      Brute force and Nash equilibrium: fin                _____
        msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)             
        
        
        for algo_name in algos:
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                        msg, algo_name)
            
            if algo_name in fct_aux.ALGO_NAMES_BF \
                or algo_name in fct_aux.ALGO_NAMES_NASH:
                
                algos_BF_NH = look4BF_NH(algos)
                    
                dico_profils_BF, dico_profils_NH, \
                dico_best_profils_BF, dico_bad_profils_BF, dico_mid_profils_BF, \
                dico_best_profils_NH, dico_bad_profils_NH, dico_mid_profils_NH \
                    = bfNhGameModel.bf_nash_game_model_1t_LOOK4BadBestMid(
                        list_dico_modes_profs_by_players_t_bestBF,
                        list_dico_modes_profs_by_players_t_badBF,
                        list_dico_modes_profs_by_players_t_midBF,
                        list_dico_modes_profs_by_players_t_bestNH,
                        list_dico_modes_profs_by_players_t_badNH,
                        list_dico_modes_profs_by_players_t_midNH,
                        keys_best_BF_NH, keys_bad_BF_NH,
                        pi_hp_plus_T, pi_hp_minus_T, 
                        m_players, t_periods,
                        arr_pl_MTvars_modif=arr_pl_MTvars_modif.copy(), t=t,
                        algos_BF_NH=algos_BF_NH,
                        pi_hp_plus=pi_hp_plus_elt, 
                        pi_hp_minus=pi_hp_minus_elt,
                        a=a, b=b,
                        gamma_version=gamma_version,
                        path_to_save=path_to_save, 
                        name_dir=name_dir, 
                        date_hhmm=date_hhmm,
                        manual_debug=manual_debug, 
                        criteria_bf=criteria_bf, dbg=dbg)
                
                nb_best_profils_bf = dico_best_profils_BF["nb_best_profils"]
                Perf_best_profils_bf = dico_best_profils_BF["Perfs"]
                nb_bad_profils_bf = dico_bad_profils_BF["nb_bad_profils"]
                Perf_bad_profils_bf = dico_bad_profils_BF["Perfs"]
                
                profils_NH = dico_profils_NH["profils"]
                nb_bad_profils_NH = dico_bad_profils_NH["nb_bad_profils"]
                Perf_bad_profils_NH = dico_bad_profils_NH["Perfs"]
                nb_best_profils_NH = dico_best_profils_NH["nb_best_profils"]
                Perf_best_profils_NH = dico_best_profils_NH["Perfs"]
                    
            elif algo_name in fct_aux.ALGO_NAMES_DET:
                # 0: Selfish-DETERMINIST, 1: Systematic-DETERMINIST
                print("*** ALGO: {} *** ".format(algo_name))
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars_DET \
                    = autoDetGameModel.determinist_balanced_player_game(
                                 arr_pl_M_T_vars_init.copy(),
                                 pi_hp_plus=pi_hp_plus_elt, 
                                 pi_hp_minus=pi_hp_minus_elt,
                                 a=a, b=b,
                                 gamma_version=gamma_version,
                                 algo_name=algo_name,
                                 used_storage=used_storage_det,
                                 path_to_save=path_to_save, 
                                 manual_debug=manual_debug, dbg=dbg)
                    
            elif algo_name == fct_aux.ALGO_NAMES_LRIx[0]:
                # 0: LRI1
                print("*** ALGO: {} *** ".format(algo_name))
                utility_function_version = 1
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                        msg, algo_name, str(learning_rate)
                                        )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars_LRI1, profils_stabilisation_LRI1, \
                k_stop_learning_LRI1, bool_equilibrium_nash_LRI1, \
                Perf_sum_Vi_LRI1 \
                    = onePeriodLriGameModel\
                        .lri_balanced_player_game_all_pijk_upper_08_onePeriod(
                            arr_pl_MTvars_init=arr_pl_MTvars_modif.copy(),
                            pi_hp_plus=pi_hp_plus_elt, 
                            pi_hp_minus=pi_hp_minus_elt,
                            a=a, b=b,
                            pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T,
                            phi_hp_plus_T=phi_hp_plus_T, phi_hp_minus_T=phi_hp_minus_T,
                            gamma_version=gamma_version,
                            k_steps=k_steps, 
                            learning_rate=learning_rate,
                            p_i_j_ks=p_i_j_ks,
                            utility_function_version=utility_function_version,
                            path_to_save=path_to_save, 
                            manual_debug=manual_debug, dbg=dbg)
                        
            elif algo_name == fct_aux.ALGO_NAMES_LRIx[1]:
                # 1: LRI2
                print("*** ALGO: {} *** ".format(algo_name))
                utility_function_version = 2
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                        msg, algo_name, str(learning_rate)
                                        )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars_LRI2, profils_stabilisation_LRI2, \
                k_stop_learning_LRI2, bool_equilibrium_nash_LRI2, \
                Perf_sum_Vi_LRI2 \
                    = onePeriodLriGameModel\
                        .lri_balanced_player_game_all_pijk_upper_08_onePeriod(
                            arr_pl_MTvars_init=arr_pl_MTvars_modif.copy(),
                            pi_hp_plus=pi_hp_plus_elt, 
                            pi_hp_minus=pi_hp_minus_elt,
                            a=a, b=b,
                            pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T,
                            phi_hp_plus_T=phi_hp_plus_T, phi_hp_minus_T=phi_hp_minus_T,
                            gamma_version=gamma_version,
                            k_steps=k_steps, 
                            learning_rate=learning_rate,
                            p_i_j_ks=p_i_j_ks,
                            utility_function_version=utility_function_version,
                            path_to_save=path_to_save, 
                            manual_debug=manual_debug, dbg=dbg) 
      
    
        print("profils_stabilisation_LRI2={}, set_profils_NH={}, profils_NH={}, nb_best_profils_bf={}".format(
                profils_stabilisation_LRI2, len(set(profils_NH)), len(profils_NH), 
                nb_best_profils_bf  ))
        
        C1 = True if len(profils_NH) > 0 else False
        C2 = True if k_stop_learning_LRI2 < k_steps else False
        C4 = k_stop_learning_LRI2 if C2 else None
        if C1 and C2:
            if tuple(profils_stabilisation_LRI2) in profils_NH:
                C3 = True
            elif tuple(profils_stabilisation_LRI2) not in profils_NH:
                C3 = False
        C5 = Perf_sum_Vi_LRI2
        C6 = Perf_best_profils_bf[0] if nb_best_profils_bf > 0 else None
        C9 = Perf_bad_profils_bf[0] if nb_bad_profils_bf > 0 else None
        if C1:
            C7 = Perf_bad_profils_NH[0] if nb_bad_profils_NH > 0 else None
            
        check_C5_inf_C6 = None
        if C5 is not None and C6 is not None and C5 <= C6:
            check_C5_inf_C6 = "OK"
        else:
            check_C5_inf_C6 = "NOK"
        check_C7_inf_C6 = None
        if C7 is not None and C6 is not None and C7 <= C6:
            check_C7_inf_C6 = "OK"
        else:
            check_C7_inf_C6 = "NOK"
                
        
        
        Cx={fct_aux.name_cols_CX["C1"]:[C1], fct_aux.name_cols_CX["C2"]:[C2], 
            fct_aux.name_cols_CX["C3"]:[C3], fct_aux.name_cols_CX["C4"]:[C4], 
            fct_aux.name_cols_CX["C5"]:[C5], fct_aux.name_cols_CX["C6"]:[C6], 
            fct_aux.name_cols_CX["C7"]:[C7], fct_aux.name_cols_CX["C9"]:[C9],
            fct_aux.name_cols_CX["check_C5_inf_C6"]:[check_C5_inf_C6], 
            fct_aux.name_cols_CX["check_C7_inf_C6"]:[check_C7_inf_C6]}
        
        path_to_save = name_dir.split(os.sep)[0:2]
        path_to_save.append("save_all_instances")
        path_to_save = os.path.join(*path_to_save)
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
            
        pd.DataFrame(Cx,index=["instance"+str(numero_instance)]).to_csv(
            os.path.join(path_to_save, "Cx_instance"+str(numero_instance)+".csv")
            )
            
    print("NB_EXECUTION cpt={}".format(cpt))
            
    
#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------
def test_execute_BF_NH_LRI_OnePeriod_N_INSTANCES_MULTI():
    # constances 
    criteria_bf = "Perf_t"
    used_storage_det = True #False #True
    manual_debug = False #True #False #True
    used_instances = False #True
    dbg = False
    
    date_hhmm="DDMM_HHMM"
    t_periods = 1 #50 #30 #35 #55 #117 #15 #3
    k_steps = 100 #250 #50000 #250 #5000 #2000 #50 #250
    NB_REPEAT_K_MAX= 3 #10 #3 #15 #30
    learning_rates = [0.01]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    fct_aux.N_DECIMALS = 8
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}, "A1.2B0.8": {"a":1.2, "b":0.8}}
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}}
    pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    pi_hp_minus = [30] #[20] #[0.33] #[15, 5]
    fct_aux.PI_0_PLUS_INIT = 4 #20 #4
    fct_aux.PI_0_MINUS_INIT = 3 #10 #3
    doc_VALUES = 24
    NB_INSTANCES = 10 #50
            
    gamma_version = -2
    
    
    prob_scen = 0.6
    prob_A_A = prob_scen; prob_A_C = 1-prob_scen;
    prob_C_A = 1-prob_scen; prob_C_C = prob_scen;
    scenario = [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
    scenario_name = "scenarioOnePeriod"
    setA_m_players_1 = 10; setC_m_players_1 = 10;                               # 20 joueurs
    setA_m_players_1 = 5; setC_m_players_1 = 5;                                 # 10 joueurs
    
    name_dir = "tests"
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    arr_pl_MTvars_init \
        = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC_doc23(
                        setA_m_players_1, setC_m_players_1, 
                        t_periods, 
                        scenario,
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl_SETAC_doc23(arr_pl_MTvars_init, 
                                                     scenario_name)
    
    pi_hp_plus_T, pi_hp_minus_T, \
    phi_hp_plus_T, phi_hp_minus_T \
        = fct_aux.compute_pi_phi_HP_minus_plus_all_t(
            arr_pl_M_T_vars_init=arr_pl_MTvars_init,
            t_periods=t_periods,
            pi_hp_plus=pi_hp_plus[0],
            pi_hp_minus=pi_hp_minus[0],
            a=dico_phiname_ab["A1B1"]["a"],
            b=dico_phiname_ab["A1B1"]["b"], 
            gamma_version=gamma_version, 
            manual_debug=manual_debug,
            dbg=dbg)
        
    algo_names = fct_aux.ALGO_NAMES_LRIx + fct_aux.ALGO_NAMES_DET \
                    + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    gamma_version = -2            
    execute_BF_NH_LRI_OnePeriod_used_Generated_N_INSTANCES_MULTI(
        arr_pl_M_T_vars_init = arr_pl_MTvars_init,
        name_dir = name_dir,
        date_hhmm = date_hhmm,
        k_steps = k_steps,
        NB_REPEAT_K_MAX = fct_aux.NB_REPEAT_K_MAX,
        algos = algo_names,
        learning_rates = learning_rates,
        pi_hp_plus = pi_hp_plus, pi_hp_minus = pi_hp_minus,
        a=dico_phiname_ab["A1B1"]["a"], 
        b=dico_phiname_ab["A1B1"]["b"],
        pi_hp_plus_T = pi_hp_plus_T, pi_hp_minus_T = pi_hp_minus_T,
        phi_hp_plus_T = phi_hp_plus_T, phi_hp_minus_T = phi_hp_minus_T,
        gamma_version=gamma_version,
        used_instances=used_instances,
        used_storage_det=used_storage_det,
        manual_debug=manual_debug, 
        criteria_bf=criteria_bf, 
        numero_instance=0,
        dbg=False)

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    boolean_execute = True #False
    
    if boolean_execute:
        test_execute_BF_NH_LRI_OnePeriod_N_INSTANCES_MULTI()
    
    print("runtime = {}".format(time.time() - ti))
