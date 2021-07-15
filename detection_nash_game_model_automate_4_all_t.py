# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:57:48 2021

@author: jwehounou
"""

import os
import time
import psutil

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import itertools as it

import force_brute_game_model_automate_4_all_t as autoBfGameModel4T

from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook


###############################################################################
#                   definition  des fonctions annexes
#
###############################################################################
# ____________          turn dico in2 df  --> debut             ______________
def turn_dico_stats_res_into_df_NASH(dico_stats_res_algo, path_to_save, 
                                     t_periods=2, 
                                     manual_debug=True, 
                                     algo_name="BEST-NASH"):
    """
    transform the dico in the row dico_nash_profils into a DataFrame

    Parameters
    ----------
    path_to_variable : TYPE
        DESCRIPTION.
        

    Returns
    -------
    None.

    """
    df = None
    for t in range(0,t_periods):
        dico_nash = dico_stats_res_algo[t]["dico_nash_profils"]
        df_t = pd.DataFrame.from_dict(dico_nash, orient='columns')
        if df is None:
            df = df_t.copy()
        else:
            df = pd.concat([df, df_t], axis=0)
            
    # save df to xlsx
    df.to_excel(os.path.join(*[path_to_save,
                               "{}_dico_NASH.xlsx".format(algo_name)]), 
                index=True)

# ____________          turn dico in2 df  --> fin               ______________

# __________         detection of NASH EQUILIBRIUM   ---> debut        ________
def detect_nash_balancing_profil(dico_profs_Vis_Perf_t, 
                                 arr_pl_M_T_vars_modif, t):
    """
    detect the profil driving to nash equilibrium
    
    dico_profs_Vis_Perf_t[tuple_prof] = dico_Vis_Pref_t with
        * tuple_prof = (S1, ...., Sm), Si is the strategie of player i
        * dico_Vis_Pref_t has keys "Pref_t" and RACINE_PLAYER+"_"+i
            * the value of "Pref_t" is \sum\limits_{1\leq i \leq N}ben_i-cst_i
            * the value of RACINE_PLAYER+"_"+i is Vi = ben_i - cst_i
            * NB : 0 <= i < m_players or  i \in [0, m_player[
    """
    nash_profils = list()
    dico_nash_profils = dict()
    cpt_nash = 0
    for key_modes_prof, dico_Vi_Pref_t in dico_profs_Vis_Perf_t.items():
        cpt_players_stables = 0
        dico_profils = dict()
        for num_pl_i, mode_i in enumerate(key_modes_prof):                      # 0 <= num_pl_i < m_player            
            Vi, ben_i, cst_i = dico_Vi_Pref_t[fct_aux.RACINE_PLAYER\
                                              +"_"+str(num_pl_i)]
            state_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                      fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
            gamma_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]
            setx = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["set"]]
            prod_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
            cons_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
            r_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
            mode_i_bar = None
            mode_i_bar = fct_aux.find_out_opposite_mode(state_i, mode_i)
            new_key_modes_prof = list(key_modes_prof)
            new_key_modes_prof[num_pl_i] = mode_i_bar
            new_key_modes_prof = tuple(new_key_modes_prof)
            
            Vi_bar = None
            Vi_bar, ben_i_bar, cst_i_bar \
                = dico_profs_Vis_Perf_t[new_key_modes_prof]\
                                          [fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)]
            if Vi >= Vi_bar:
                cpt_players_stables += 1
            dico_profils[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)+"_t_"+str(t)] \
                = {"set":setx, "state":state_i, "mode_i":mode_i, "Vi":Vi, 
                   "gamma_i":gamma_i, "prod":prod_i, "cons":cons_i, "r_i":r_i,
                   "ben":ben_i, "cst":cst_i}
        
        dico_profils["mode_profil"] = key_modes_prof  
            
        if cpt_players_stables == len(key_modes_prof):
            nash_profils.append(key_modes_prof)
            Perf_t = dico_profs_Vis_Perf_t[key_modes_prof]["Perf_t"]
            dico_profils["Perf_t"] = Perf_t
            dico_nash_profils["NASH_"+str(cpt_nash)] = (dico_profils)
            cpt_nash += 1
                
    return nash_profils, dico_nash_profils
# __________         detection of NASH EQUILIBRIUM   ---> fin          ________


# __________       main function of NASH EQUILIBRIUM   ---> debut      ________
def nash_balanced_player_game(arr_pl_M_T_vars_init,
                                pi_hp_plus=0.2, 
                                pi_hp_minus=0.33,
                                a=1, b=1,
                                gamma_version=1,
                                path_to_save="tests", 
                                name_dir="tests", 
                                date_hhmm="DDMM_HHMM",
                                manual_debug=False, 
                                dbg=False):
    
    print("\n \n game: {}, pi_hp_minus ={} ---> debut \n"\
          .format(pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,))                                # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus_T = np.empty(shape=(t_periods,))                                  # shape (T_PERIODS,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_0_minus_T.fill(np.nan)
    pi_hp_plus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_hp_plus_T.fill(np.nan)
    pi_hp_minus_T = np.empty(shape=(t_periods,))                                # shape (T_PERIODS,)
    pi_hp_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,))                                      # shape (T_PERIODS,)
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods))                           # shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    CC_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    CC_is_M.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    BB_is_M.fill(np.nan)
    RU_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    RU_is_M.fill(np.nan)    
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    dico_id_players = {"players":[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i) 
                                  for num_pl_i in range(0, m_players)]}
    df_nash = pd.DataFrame.from_dict(dico_id_players)
    
    # _________  creation des arrays pour chaque algo  _______________________
    arr_pl_M_T_vars_modif_BADN = None
    arr_pl_M_T_vars_modif_BESTN = None
    arr_pl_M_T_vars_modif_MIDN = None
    
    pi_sg_plus_T_BESTN = pi_sg_plus_T.copy()
    pi_sg_minus_T_BESTN = pi_sg_minus_T.copy()
    pi_0_plus_T_BESTN = pi_0_plus_T.copy()
    pi_0_minus_T_BESTN = pi_0_minus_T.copy()
    B_is_M_BESTN = B_is_M.copy()
    C_is_M_BESTN = C_is_M.copy()
    b0_ts_T_BESTN = b0_ts_T.copy()
    c0_ts_T_BESTN = c0_ts_T.copy()
    BENs_M_T_BESTN = BENs_M_T.copy()
    CSTs_M_T_BESTN = CSTs_M_T.copy()
    CC_is_M_BESTN = CC_is_M.copy()
    BB_is_M_BESTN = BB_is_M.copy()
    RU_is_M_BESTN = RU_is_M.copy()
    C_is_M_BESTN = CC_is_M.copy()
    B_is_M_BESTN = BB_is_M.copy()
    dico_stats_res_BESTN = dict()
    df_nash_BESTN = df_nash.copy()
    
    pi_sg_plus_T_BADN = pi_sg_plus_T.copy()
    pi_sg_minus_T_BADN = pi_sg_minus_T.copy()
    pi_0_plus_T_BADN = pi_0_plus_T.copy()
    pi_0_minus_T_BADN = pi_0_minus_T.copy()
    B_is_M_BADN = B_is_M.copy()
    C_is_M_BADN = C_is_M.copy()
    b0_ts_T_BADN = b0_ts_T.copy()
    c0_ts_T_BADN = c0_ts_T.copy()
    BENs_M_T_BADN = BENs_M_T.copy()
    CSTs_M_T_BADN = CSTs_M_T.copy()
    CC_is_M_BADN = CC_is_M.copy()
    BB_is_M_BADN = BB_is_M.copy()
    RU_is_M_BADN = RU_is_M.copy()
    C_is_M_BADN = CC_is_M.copy()
    B_is_M_BADN = BB_is_M.copy()
    dico_stats_res_BADN = dict()
    df_nash_BADN = df_nash.copy()
    
    pi_sg_plus_T_MIDN = pi_sg_plus_T.copy()
    pi_sg_minus_T_MIDN = pi_sg_minus_T.copy()
    pi_0_plus_T_MIDN = pi_0_plus_T.copy()
    pi_0_minus_T_MIDN = pi_0_minus_T.copy()
    B_is_M_MIDN = B_is_M.copy()
    C_is_M_MIDN = C_is_M.copy()
    b0_ts_T_MIDN = b0_ts_T.copy()
    c0_ts_T_MIDN = c0_ts_T.copy()
    BENs_M_T_MIDN = BENs_M_T.copy()
    CSTs_M_T_MIDN = CSTs_M_T.copy()
    CC_is_M_MIDN = CC_is_M.copy()
    BB_is_M_MIDN = BB_is_M.copy()
    RU_is_M_MIDN = RU_is_M.copy()
    C_is_M_MIDN = CC_is_M.copy()
    B_is_M_MIDN = BB_is_M.copy()
    dico_stats_res_MIDN = dict()
    df_nash_MIDN = df_nash.copy()
    
    arr_pl_M_T_vars_modif_BADN = arr_pl_M_T_vars_modif.copy()
    arr_pl_M_T_vars_modif_BESTN = arr_pl_M_T_vars_modif.copy()
    arr_pl_M_T_vars_modif_MIDN = arr_pl_M_T_vars_modif.copy()
    
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    dico_mode_prof_by_players_T = dict()
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        else:
            pi_sg_plus_t_minus_1 = pi_sg_plus_t0_minus_1 if t == 0 \
                                                         else pi_sg_plus_t
            pi_sg_minus_t_minus_1 = pi_sg_minus_t0_minus_1 if t == 0 \
                                                            else pi_sg_minus_t
            
            q_t_minus, q_t_plus = fct_aux.compute_upper_bound_quantity_energy(
                                    arr_pl_M_T_vars_modif, t)
            phi_hp_minus_t = fct_aux.compute_cost_energy_bought_by_SG_2_HP(
                                pi_hp_minus=pi_hp_minus, 
                                quantity=q_t_minus,
                                b=b)
            phi_hp_plus_t = fct_aux.compute_benefit_energy_sold_by_SG_2_HP(
                                pi_hp_plus=pi_hp_plus, 
                                quantity=q_t_plus,
                                a=a)
            pi_hp_minus_t = round(phi_hp_minus_t/q_t_minus, fct_aux.N_DECIMALS) \
                            if q_t_minus != 0 \
                            else 0
            pi_hp_plus_t = round(phi_hp_plus_t/q_t_plus, fct_aux.N_DECIMALS) \
                            if q_t_plus != 0 \
                            else 0
                            
            pi_0_plus_t = None
            if pi_hp_minus_t > 0:
                pi_0_plus_t = round(pi_sg_minus_t_minus_1*pi_hp_plus_t/pi_hp_minus_t, 
                                    fct_aux.N_DECIMALS)
            else:
                pi_0_plus_t = 0
            pi_0_minus_t = pi_sg_minus_t_minus_1
            if t == 0:
               pi_0_plus_t = fct_aux.PI_0_PLUS_INIT #4
               pi_0_minus_t = fct_aux.PI_0_MINUS_INIT #3
               
        arr_pl_M_T_vars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus_t=pi_hp_plus_t, pi_hp_minus_t=pi_hp_minus_t,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)
        print('gamma_is={}'.format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]))
        
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))
        
        arr_pl_M_T_vars_modif_BADN[:,t,:] = arr_pl_M_T_vars_modif[:,t,:]
        arr_pl_M_T_vars_modif_BESTN[:,t,:] = arr_pl_M_T_vars_modif[:,t,:]
        arr_pl_M_T_vars_modif_MIDN[:,t,:] = arr_pl_M_T_vars_modif[:,t,:]
            
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        pi_hp_plus_T[t] = pi_hp_plus_t
        pi_hp_minus_T[t] = pi_hp_minus_t
        
        
        # balanced player game at instant t    
        dico_profs_Vis_Perf_t = dict()
        cpt_profs = 0
        
        mode_profiles = it.product(*possibles_modes)
        for mode_profile in mode_profiles:
            
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus, 
                    a, b,
                    pi_0_plus_t, pi_0_minus_t,
                    random_mode=False,
                    manual_debug=manual_debug, dbg=False
                    )
            
            In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_mode_prof, 
                                t)
            
            bens_csts_t = bens_t - csts_t
            Perf_t = np.sum(bens_csts_t, axis=0)
            
            for num_pl_i in range(0, m_players):
                prod_i = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]];
                cons_i = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]];
                r_i = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]];
                Pi = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]];
                Ci = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]];
                Si_max = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]];
                Si = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]];
                Si_old = arr_pl_M_T_vars_mode_prof[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]];
            
                print("prod_i={}, cons_i={}, r_i={}, Pi={}, Ci={}, Si_max={}, Si={}, Si_old={}".format(prod_i,cons_i,r_i,Pi,Ci,Si_max,Si,Si_old))
            
            print("pi_0_plus_t={}, pi_0_minus_t={}, ".format(pi_0_plus_t, pi_0_minus_t))
            print("b0_t={}, c0_t={}, bens_t={}, csts_t={},".format(b0_t, c0_t, bens_t, csts_t))
            print("mode_profile={}, bens_csts={}, Perf_t={}".format(mode_profile, bens_csts_t, Perf_t))
            
            
            
            dico_Vis_Pref_t = dict()
            for num_pl_i in range(bens_csts_t.shape[0]):                        # bens_csts_t.shape[0] = m_players
                dico_Vis_Pref_t[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)] \
                    = (bens_csts_t[num_pl_i], 
                       bens_t[num_pl_i], 
                       csts_t[num_pl_i])
                    
                dico_vars = dict()
                dico_vars["Vi"] = round(bens_csts_t[num_pl_i], 2)
                dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
                dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
                variables = ["set", "state_i", "mode_i", "Pi", "Ci", "Si_max", 
                             "Si_old", "Si", "prod_i", "cons_i", "r_i", 
                             "Si_minus", "Si_plus", "gamma_i"]
                for variable in variables:
                    dico_vars[variable] = arr_pl_M_T_vars_mode_prof[
                                            num_pl_i, t, 
                                            fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                    
                dico_mode_prof_by_players_T["PLAYER_"
                                            +str(num_pl_i)
                                            +"_t_"+str(t)
                                            +"_"+str(cpt_profs)] \
                    = dico_vars
                    
            dico_mode_prof_by_players_T["Perf_t_"+str(t)+"_"+str(cpt_profs)] = round(Perf_t, 2)
            dico_mode_prof_by_players_T["b0_t_"+str(t)+"_"+str(cpt_profs)] = round(b0_t,2)
            dico_mode_prof_by_players_T["c0_t_"+str(t)+"_"+str(cpt_profs)] = round(c0_t,2)
            dico_mode_prof_by_players_T["Out_sg_"+str(t)+"_"+str(cpt_profs)] = round(Out_sg,2)
            dico_mode_prof_by_players_T["In_sg_"+str(t)+"_"+str(cpt_profs)] = round(In_sg,2)
            dico_mode_prof_by_players_T["pi_sg_plus_t_"+str(t)+"_"+str(cpt_profs)] = round(pi_sg_plus_t,2)
            dico_mode_prof_by_players_T["pi_sg_minus_t_"+str(t)+"_"+str(cpt_profs)] = round(pi_sg_minus_t,2)
            dico_mode_prof_by_players_T["pi_0_plus_t_"+str(t)+"_"+str(cpt_profs)] = round(pi_0_plus_t,2)
            dico_mode_prof_by_players_T["pi_0_minus_t_"+str(t)+"_"+str(cpt_profs)] = round(pi_0_minus_t,2)
            
                    
            dico_Vis_Pref_t["Perf_t"] = Perf_t
            dico_Vis_Pref_t["b0"] = b0_t
            dico_Vis_Pref_t["c0"] = c0_t
            
            dico_profs_Vis_Perf_t[mode_profile] = dico_Vis_Pref_t
            cpt_profs += 1
            
            if cpt_profs%5000 == 0:
                print("cpt_prof={}".format(cpt_profs))
                
        # detection of NASH profils
        nash_profils = list();
        dico_nash_profils = list()
        nash_profils, dico_nash_profils = detect_nash_balancing_profil(
                        dico_profs_Vis_Perf_t,
                        arr_pl_M_T_vars_modif, 
                        t)
        
        # delete all occurences of the profiles 
        print("----> avant supp doublons nash_profils={}".format(len(nash_profils)))
        nash_profils = set(nash_profils)
        print("----> apres supp doublons nash_profils={}".format(len(nash_profils)))
        
        # create dico of nash profils with key is Pref_t and value is profil
        dico_Perft_nashProfil = dict()
        for nash_mode_profil in nash_profils:
            Perf_t = dico_profs_Vis_Perf_t[nash_mode_profil]["Perf_t"]
            if Perf_t in dico_Perft_nashProfil:
                dico_Perft_nashProfil[Perf_t].append(nash_mode_profil)
            else:
                dico_Perft_nashProfil[Perf_t] = [nash_mode_profil]
                
        if len(dico_Perft_nashProfil.keys()) != 0:        
            arr_pl_M_T_vars_modif_algo = None
            b0_ts_T_algo, c0_ts_T_algo = None, None
            BENs_M_T_algo, CSTs_M_T_algo = None, None
            pi_0_plus_T_algo, pi_0_minus_T_algo = None, None
            df_nash_algo = None
            for algo_name in fct_aux.ALGO_NAMES_NASH:
                
                if algo_name == fct_aux.ALGO_NAMES_NASH[0]:                     # BEST-NASH
                    arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_BESTN.copy()
                    pi_sg_plus_T_algo = pi_sg_plus_T_BESTN.copy()
                    pi_sg_minus_T_algo = pi_sg_minus_T_BESTN.copy()
                    b0_ts_T_algo = b0_ts_T_BESTN.copy()
                    c0_ts_T_algo = c0_ts_T_BESTN.copy()
                    BENs_M_T_algo = BENs_M_T_BESTN.copy()
                    CSTs_M_T_algo = CSTs_M_T_BESTN.copy()
                    pi_0_plus_T_algo = pi_0_plus_T_BESTN.copy() 
                    pi_0_minus_T_algo = pi_0_minus_T_BESTN.copy()
                    dico_stats_res_algo = dico_stats_res_BESTN.copy()
                    df_nash_algo = df_nash_BESTN.copy()
                                   
                elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                   # BAD-NASH
                    arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_BADN.copy()
                    pi_sg_plus_T_algo = pi_sg_plus_T_BADN.copy()
                    pi_sg_minus_T_algo = pi_sg_minus_T_BADN.copy()
                    b0_ts_T_algo = b0_ts_T_BADN.copy()
                    c0_ts_T_algo = c0_ts_T_BADN.copy()
                    BENs_M_T_algo = BENs_M_T_BADN.copy()
                    CSTs_M_T_algo = CSTs_M_T_BADN.copy()
                    pi_0_plus_T_algo = pi_0_plus_T_BADN.copy()
                    pi_0_minus_T_algo = pi_0_minus_T_BADN.copy()
                    dico_stats_res_algo = dico_stats_res_BADN.copy()
                    df_nash_algo = df_nash_BADN.copy()
                    
                elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                   # MIDDLE-NASH
                    arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_MIDN.copy()
                    pi_sg_plus_T_algo = pi_sg_plus_T_MIDN.copy()
                    pi_sg_minus_T_algo = pi_sg_minus_T_MIDN.copy()
                    b0_ts_T_algo = b0_ts_T_MIDN.copy()
                    c0_ts_T_algo = c0_ts_T_MIDN.copy()
                    BENs_M_T_algo = BENs_M_T_MIDN.copy()
                    CSTs_M_T_algo = CSTs_M_T_MIDN.copy()
                    pi_0_plus_T_algo = pi_0_plus_T_MIDN.copy()
                    pi_0_minus_T_algo = pi_0_minus_T_MIDN.copy()
                    dico_stats_res_algo = dico_stats_res_MIDN.copy()
                    df_nash_algo = df_nash_MIDN.copy()
                
                # ____      game beginning for one algo for t ---> debut      _____
                # min, max, mean of Perf_t
                print("algo_name = {}".format(algo_name))
                best_key_Perf_t = None
                if algo_name == fct_aux.ALGO_NAMES_NASH[0]:                     # BEST-NASH
                    best_key_Perf_t = max(dico_Perft_nashProfil.keys())
                elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                   # BAD-NASH
                    best_key_Perf_t = min(dico_Perft_nashProfil.keys())
                elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                   # MIDDLE-NASH
                    mean_key_Perf_t  = np.mean(list(dico_Perft_nashProfil.keys()))
                    if mean_key_Perf_t in dico_Perft_nashProfil.keys():
                        best_key_Perf_t = mean_key_Perf_t
                    else:
                        sorted_keys = sorted(dico_Perft_nashProfil.keys())
                        boolean = True; i_key = 1
                        while boolean:
                            if sorted_keys[i_key] <= mean_key_Perf_t:
                                i_key += 1
                            else:
                                boolean = False; i_key -= 1
                        best_key_Perf_t = sorted_keys[i_key]
                        
                
                # find the best, bad, middle key in dico_Perft_nashProfil and 
                # the best, bad, middle nash_mode_profile
                best_nash_mode_profiles = dico_Perft_nashProfil[best_key_Perf_t]
                best_nash_mode_profile = None
                if len(best_nash_mode_profiles) == 1:
                    best_nash_mode_profile = best_nash_mode_profiles[0]
                else:
                    rd = np.random.randint(0, len(best_nash_mode_profiles))
                    best_nash_mode_profile = best_nash_mode_profiles[rd]
                
                print("** Running at t={}: numbers of -> cpt_profils={}, nash_profils={}, {}_nash_mode_profiles={}, nbre_cle_dico_Perft_nashProf={}"\
                      .format(t, cpt_profs, 
                        len(nash_profils),
                        algo_name.split("-")[0], len(best_nash_mode_profiles),
                        list(dico_Perft_nashProfil.keys())      
                              ))
                print("{}_key_Perf_t={}, {}_nash_mode_profile={}".format(
                        algo_name.split("-")[0], best_key_Perf_t, 
                        algo_name.split("-")[0], 
                        best_nash_mode_profile))
                
                arr_pl_M_T_vars_nash_mode_prof_algo, \
                b0_t, c0_t, \
                bens_t, csts_t, \
                pi_sg_plus_t, pi_sg_minus_t, \
                dico_gamme_t_nash_mode_prof \
                    = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                        arr_pl_M_T_vars_modif_algo.copy(),
                        best_nash_mode_profile, t,
                        pi_hp_plus, pi_hp_minus, 
                        a=a, b=b,
                        pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                        random_mode=False,
                        manual_debug=manual_debug, dbg=dbg
                        )
                    
                suff = algo_name.split("-")[0]
                dico_stats_res_algo[t] = {"gamma_i": dico_gamme_t_nash_mode_prof,
                                "nash_profils": nash_profils,
                                "nb_nash_profils": len(nash_profils),
                                suff+"_nash_profil": best_nash_mode_profile,
                                suff+"_Perf_t": best_key_Perf_t,
                                "nb_nash_profil_byPerf_t": 
                                    len(dico_Perft_nashProfil[best_key_Perf_t]),
                                "dico_nash_profils": dico_nash_profils,
                                suff+"_b0_t": b0_t,
                                suff+"_c0_t": c0_t
                                }
                    
                # pi_sg_{plus,minus} of shape (T_PERIODS,)
                if np.isnan(pi_sg_plus_t):
                    pi_sg_plus_t = 0
                if np.isnan(pi_sg_minus_t):
                    pi_sg_minus_t = 0
                pi_sg_plus_T_algo[t] = pi_sg_plus_t
                pi_sg_minus_T_algo[t] = pi_sg_minus_t
                pi_0_plus_T_algo[t] = pi_0_plus_t
                pi_0_minus_T_algo[t] = pi_0_minus_t
                
                # b0_ts, c0_ts of shape (T_PERIODS,)
                b0_ts_T_algo[t] = b0_t
                c0_ts_T_algo[t] = c0_t
                
                # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
                BENs_M_T_algo[:,t] = bens_t
                CSTs_M_T_algo[:,t] = csts_t
                
                # checkout NASH equilibrium
                bens_csts_M_t = bens_t - csts_t
                df_nash_t = None        
                df_nash_t = fct_aux.checkout_nash_4_profils_by_periods(
                                arr_pl_M_T_vars_nash_mode_prof_algo.copy(),
                                arr_pl_M_T_vars_init,
                                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                                a=a, b=b,
                                pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                                bens_csts_M_t=bens_csts_M_t,
                                t=t,
                                manual_debug=manual_debug)
                df_nash_algo = pd.merge(df_nash_algo, df_nash_t, 
                                        on='players', 
                                        how='outer')
                # ____      game beginning for one algo for t ---> FIN        _____
                
                # __________        compute prices variables         ____________________
                # B_is, C_is of shape (M_PLAYERS, )
                prod_i_M_T_algo = arr_pl_M_T_vars_nash_mode_prof_algo[
                                        :,:t_periods, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
                cons_i_M_T_algo = arr_pl_M_T_vars_nash_mode_prof_algo[
                                        :,:t_periods, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
                B_is_M_algo = np.sum(b0_ts_T_algo * prod_i_M_T_algo, axis=1)
                C_is_M_algo = np.sum(c0_ts_T_algo * cons_i_M_T_algo, axis=1)
                
                # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
                CONS_is_M_algo = np.sum(cons_i_M_T_algo, axis=1)
                PROD_is_M_algo = np.sum(prod_i_M_T_algo, axis=1)
                BB_is_M_algo = pi_sg_plus_T_algo[t] * PROD_is_M_algo #np.sum(PROD_is)
                for num_pl, bb_i in enumerate(BB_is_M_algo):
                    if bb_i != 0:
                        print("player {}, BB_i={}".format(num_pl, bb_i))
                    if np.isnan(bb_i):
                        print("player {},PROD_is={}, pi_sg={}".format(num_pl, 
                              PROD_is_M_algo[num_pl], pi_sg_plus_T_algo[t]))
                CC_is_M_algo = pi_sg_minus_T_algo[t] * CONS_is_M_algo #np.sum(CONS_is)
                RU_is_M_algo = BB_is_M_algo - CC_is_M_algo
                print("{}: t={}, pi_sg_plus_T={}, pi_sg_minus_T={} \n".format(
                    algo_name, t, pi_sg_plus_T_algo[t], pi_sg_minus_T_algo[t] ))
                
                # __________        compute prices variables         ____________________
                
                # __________        maj arrays for all algos: debut    ____________
                if algo_name == fct_aux.ALGO_NAMES_NASH[0]:                           # BEST-BRUTE-FORCE
                    arr_pl_M_T_vars_modif_BESTN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                    pi_sg_plus_T_BESTN[t] = pi_sg_plus_T_algo[t]
                    pi_sg_minus_T_BESTN[t] = pi_sg_minus_T_algo[t]
                    pi_0_plus_T_BESTN[t] = pi_0_plus_T_algo[t]
                    pi_0_minus_T_BESTN[t] = pi_0_minus_T_algo[t]
                    B_is_M_BESTN = B_is_M_algo.copy()
                    C_is_M_BESTN = C_is_M_algo.copy()
                    b0_ts_T_BESTN = b0_ts_T_algo.copy()
                    c0_ts_T_BESTN = c0_ts_T_algo.copy()
                    BENs_M_T_BESTN = BENs_M_T_algo.copy()
                    CSTs_M_T_BESTN = CSTs_M_T_algo.copy()
                    CC_is_M_BESTN = CC_is_M_algo.copy()
                    BB_is_M_BESTN = BB_is_M_algo.copy()
                    RU_is_M_BESTN = RU_is_M_algo.copy()
                    dico_stats_res_BESTN = dico_stats_res_algo.copy()
                    df_nash_BESTN = df_nash_algo.copy()
                    
                elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                         # BAD-BRUTE-FORCE
                    arr_pl_M_T_vars_modif_BADN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                    pi_sg_plus_T_BADN[t] = pi_sg_plus_T_algo[t]
                    pi_sg_minus_T_BADN[t] = pi_sg_minus_T_algo[t]
                    pi_0_plus_T_BADN[t] = pi_0_plus_T_algo[t]
                    pi_0_minus_T_BADN[t] = pi_0_minus_T_algo[t]
                    B_is_M_BADN = B_is_M_algo.copy()
                    C_is_M_BADN = C_is_M_algo.copy()
                    b0_ts_T_BADN = b0_ts_T_algo.copy()
                    c0_ts_T_BADN = c0_ts_T_algo.copy()
                    BENs_M_T_BADN = BENs_M_T_algo.copy()
                    CSTs_M_T_BADN = CSTs_M_T_algo.copy()
                    CC_is_M_BADN = CC_is_M_algo.copy()
                    BB_is_M_BADN = BB_is_M_algo.copy()
                    RU_is_M_BADN = RU_is_M_algo.copy()
                    dico_stats_res_BADN = dico_stats_res_algo.copy()
                    df_nash_BADN = df_nash_algo.copy()
                    
                elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                         # MIDDLE-BRUTE-FORCE
                    arr_pl_M_T_vars_modif_MIDN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                    pi_sg_plus_T_MIDN[t] = pi_sg_plus_T_algo[t]
                    pi_sg_minus_T_MIDN[t] = pi_sg_minus_T_algo[t]
                    pi_0_plus_T_MIDN[t] = pi_0_plus_T_algo[t]
                    pi_0_minus_T_MIDN[t] = pi_0_minus_T_algo[t]
                    B_is_M_MIDN = B_is_M_algo.copy()
                    C_is_M_MIDN = C_is_M_algo.copy()
                    b0_ts_T_MIDN = b0_ts_T_algo.copy()
                    c0_ts_T_MIDN = c0_ts_T_algo.copy()
                    BENs_M_T_MIDN = BENs_M_T_algo.copy()
                    CSTs_M_T_MIDN = CSTs_M_T_algo.copy()
                    CC_is_M_MIDN = CC_is_M_algo.copy()
                    BB_is_M_MIDN = BB_is_M_algo.copy()
                    RU_is_M_MIDN = RU_is_M_algo.copy()
                    dico_stats_res_MIDN = dico_stats_res_algo.copy()
                    df_nash_MIDN = df_nash_algo.copy()
                
                # __________        maj arrays for all algos: fin      ____________
        else:
            if algo_name == fct_aux.ALGO_NAMES_NASH[0]:                           # BEST-BRUTE-FORCE
                # arr_pl_M_T_vars_modif_BESTN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                pi_sg_plus_T_BESTN[t] = 0
                pi_sg_minus_T_BESTN[t] = 0
                pi_0_plus_T_BESTN[t] = 0
                pi_0_minus_T_BESTN[t] = 0
                B_is_M_BESTN[:] = 0
                C_is_M_BESTN[:] = 0
                b0_ts_T_BESTN[t] = 0
                c0_ts_T_BESTN[t] = 0
                BENs_M_T_BESTN[:,t] = 0
                CSTs_M_T_BESTN[:,t] = 0
                CC_is_M_BESTN[:] = 0
                BB_is_M_BESTN[:] = 0
                RU_is_M_BESTN[:] = 0
                dico_stats_res_BESTN[t] = {"gamma_i": dict(),
                                        "nash_profils": list(),
                                        "nb_nash_profils": 0,
                                        suff+"_nash_profil": (),
                                        suff+"_Perf_t": 0,
                                        "nb_nash_profil_byPerf_t": 0,
                                        "dico_nash_profils": dict(),
                                        suff+"_b0_t": 0,
                                        suff+"_c0_t": 0
                                        }
                
            elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                         # BAD-BRUTE-FORCE
                # arr_pl_M_T_vars_modif_BADN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                pi_sg_plus_T_BADN[t] = 0
                pi_sg_minus_T_BADN[t] = 0
                pi_0_plus_T_BADN[t] = 0
                pi_0_minus_T_BADN[t] = 0
                B_is_M_BADN[:] = 0
                C_is_M_BADN[:] = 0
                b0_ts_T_BADN[t] = 0
                c0_ts_T_BADN[t] = 0
                BENs_M_T_BADN[:,t] = 0
                CSTs_M_T_BADN[:,t] = 0
                CC_is_M_BADN[:] = 0
                BB_is_M_BADN[:] = 0
                RU_is_M_BADN[:] = 0
                dico_stats_res_BADN[t] = {"gamma_i": dict(),
                                        "nash_profils": list(),
                                        "nb_nash_profils": 0,
                                        suff+"_nash_profil": (),
                                        suff+"_Perf_t": 0,
                                        "nb_nash_profil_byPerf_t": 0,
                                        "dico_nash_profils": dict(),
                                        suff+"_b0_t": 0,
                                        suff+"_c0_t": 0
                                        }
                
            elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                         # MIDDLE-BRUTE-FORCE
                # arr_pl_M_T_vars_modif_MIDN = arr_pl_M_T_vars_nash_mode_prof_algo.copy()
                pi_sg_plus_T_MIDN[t] = 0
                pi_sg_minus_T_MIDN[t] = 0
                pi_0_plus_T_MIDN[t] = 0
                pi_0_minus_T_MIDN[t] = 0
                B_is_M_MIDN[:] = 0
                C_is_M_MIDN[:] = 0
                b0_ts_T_MIDN[t] = 0
                c0_ts_T_MIDN[t] = 0
                BENs_M_T_MIDN[:,t] = 0
                CSTs_M_T_MIDN[:,t] = 0
                CC_is_M_MIDN[:] = 0
                BB_is_M_MIDN[:] = 0
                RU_is_M_MIDN[:] = 0
                dico_stats_res_MIDN[t] = {"gamma_i": dict(),
                                        "nash_profils": list(),
                                        "nb_nash_profils": 0,
                                        suff+"_nash_profil": (),
                                        suff+"_Perf_t": 0,
                                        "nb_nash_profil_byPerf_t": 0,
                                        "dico_nash_profils": dict(),
                                        suff+"_b0_t": 0,
                                        suff+"_c0_t": 0
                                        }
                    
    # ____      game beginning for all t_period ---> fin      _____
    
    #_______      save computed variables locally from algo_name     __________
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    
    print("path_to_save={}".format(path_to_save))
    algo_name = fct_aux.ALGO_NAMES_NASH[0]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash_BESTN.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)
    if arr_pl_M_T_vars_modif_BESTN.shape[0] < 11:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BESTN, 
                   b0_ts_T_BESTN, c0_ts_T_BESTN, B_is_M_BESTN, C_is_M_BESTN, 
                   BENs_M_T_BESTN, CSTs_M_T_BESTN, 
                   BB_is_M_BESTN, CC_is_M_BESTN, RU_is_M_BESTN, 
                   pi_sg_minus_T_BESTN, pi_sg_plus_T_BESTN, 
                   pi_0_minus_T_BESTN, pi_0_plus_T_BESTN,
                   pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_BESTN, 
                   algo=algo_name, 
                   dico_best_steps=dict())
    else:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BESTN, 
                    b0_ts_T_BESTN, c0_ts_T_BESTN, B_is_M_BESTN, C_is_M_BESTN, 
                    BENs_M_T_BESTN, CSTs_M_T_BESTN, 
                    BB_is_M_BESTN, CC_is_M_BESTN, RU_is_M_BESTN, 
                    pi_sg_minus_T_BESTN, pi_sg_plus_T_BESTN, 
                    pi_0_minus_T_BESTN, pi_0_plus_T_BESTN,
                    pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_BESTN, 
                    algo=algo_name, 
                    dico_best_steps=dict())
    turn_dico_stats_res_into_df_NASH(dico_stats_res_algo=dico_stats_res_BESTN, 
                                path_to_save=path_to_save, 
                                t_periods=t_periods, 
                                manual_debug=manual_debug, 
                                algo_name=algo_name) \
        if len(dico_Perft_nashProfil.keys()) != 0 \
        else None
    
    algo_name = fct_aux.ALGO_NAMES_NASH[1]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash_BADN.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)
    if arr_pl_M_T_vars_modif_BADN.shape[0] < 11:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BADN, 
                   b0_ts_T_BADN, c0_ts_T_BADN, B_is_M_BADN, C_is_M_BADN, 
                   BENs_M_T_BADN, CSTs_M_T_BADN, 
                   BB_is_M_BADN, CC_is_M_BADN, RU_is_M_BADN, 
                   pi_sg_minus_T_BADN, pi_sg_plus_T_BADN, 
                   pi_0_minus_T_BADN, pi_0_plus_T_BADN,
                   pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_BADN, 
                   algo=algo_name,
                   dico_best_steps=dict())
    else:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BADN, 
                    b0_ts_T_BADN, c0_ts_T_BADN, B_is_M_BADN, C_is_M_BADN, 
                    BENs_M_T_BADN, CSTs_M_T_BADN, 
                    BB_is_M_BADN, CC_is_M_BADN, RU_is_M_BADN, 
                    pi_sg_minus_T_BADN, pi_sg_plus_T_BADN, 
                    pi_0_minus_T_BADN, pi_0_plus_T_BADN,
                    pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_BADN, 
                    algo=algo_name,
                    dico_best_steps=dict())  
    turn_dico_stats_res_into_df_NASH(dico_stats_res_algo=dico_stats_res_BADN, 
                                path_to_save=path_to_save, 
                                t_periods=t_periods, 
                                manual_debug=manual_debug, 
                                algo_name=algo_name) \
        if len(dico_Perft_nashProfil.keys()) != 0 \
        else None
    
    algo_name = fct_aux.ALGO_NAMES_NASH[2]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash_MIDN.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)
    if arr_pl_M_T_vars_modif_MIDN.shape[0] < 11: 
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_MIDN, 
                   b0_ts_T_MIDN, c0_ts_T_MIDN, B_is_M_MIDN, C_is_M_MIDN, 
                   BENs_M_T_MIDN, CSTs_M_T_MIDN, 
                   BB_is_M_MIDN, CC_is_M_MIDN, RU_is_M_MIDN, 
                   pi_sg_minus_T_MIDN, pi_sg_plus_T_MIDN, 
                   pi_0_minus_T_MIDN, pi_0_plus_T_MIDN,
                   pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_MIDN, 
                   algo=algo_name,
                   dico_best_steps=dict())
    else:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_MIDN, 
                    b0_ts_T_MIDN, c0_ts_T_MIDN, B_is_M_MIDN, C_is_M_MIDN, 
                    BENs_M_T_MIDN, CSTs_M_T_MIDN, 
                    BB_is_M_MIDN, CC_is_M_MIDN, RU_is_M_MIDN, 
                    pi_sg_minus_T_MIDN, pi_sg_plus_T_MIDN, 
                    pi_0_minus_T_MIDN, pi_0_plus_T_MIDN,
                    pi_hp_plus_T, pi_hp_minus_T, dico_stats_res_MIDN, 
                    algo=algo_name,
                    dico_best_steps=dict())
    turn_dico_stats_res_into_df_NASH(dico_stats_res_algo=dico_stats_res_MIDN, 
                                path_to_save=path_to_save, 
                                t_periods=t_periods, 
                                manual_debug=manual_debug, 
                                algo_name=algo_name) \
        if len(dico_Perft_nashProfil.keys()) != 0 \
        else None
        
    return arr_pl_M_T_vars_modif
        
# __________       main function of NASH EQUILIBRIUM   ---> fin        ________

# ______       generate mode profile and its Perf_ts   ---> debut        ______
def generer_mode_profiles(mode_profiles,
                          arr_pl_M_T_vars_modif,
                          t,
                          pi_hp_plus, pi_hp_minus, 
                          a, b,
                          pi_0_plus_t, pi_0_minus_t,
                          random_mode,
                          manual_debug, dbg):
    m_players = arr_pl_M_T_vars_modif.shape[0]
    dico_profs_Vis_Perf_t = dict(); dico_mode_prof_by_players = dict()
    cpt_profs = 0
    for mode_profile in mode_profiles:
        
        dico_gamme_t = dict()
        arr_pl_M_T_vars_mode_prof, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t \
            = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                arr_pl_M_T_vars_modif.copy(),
                mode_profile, t,
                pi_hp_plus, pi_hp_minus, 
                a, b,
                pi_0_plus_t, pi_0_minus_t,
                random_mode=False,
                manual_debug=manual_debug, dbg=False
                )
        
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                            arr_pl_M_T_vars_mode_prof, 
                            t)
        
        bens_csts_t = bens_t - csts_t
        Perf_t = np.sum(bens_csts_t, axis=0)
        
        dico_Vis_Pref_t = dict()
        for num_pl_i in range(0, m_players):
            dico_Vis_Pref_t[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)] \
                = (bens_csts_t[num_pl_i], 
                   bens_t[num_pl_i], 
                   csts_t[num_pl_i])
                
            dico_vars = dict()
            dico_vars["Vi"] = round(bens_csts_t[num_pl_i], 2)
            dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
            dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
            variables = ["set", "state_i", "mode_i", "Pi", "Ci", "Si_max", 
                         "Si_old", "Si", "prod_i", "cons_i", "r_i", 
                         "Si_minus", "Si_plus", "gamma_i"]
            for variable in variables:
                dico_vars[variable] = arr_pl_M_T_vars_mode_prof[
                                        num_pl_i, t, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                
            dico_mode_prof_by_players[fct_aux.RACINE_PLAYER
                                        +str(num_pl_i)
                                        +"_t_"+str(t)
                                        +"_"+str(cpt_profs)] \
                = dico_vars
                
        dico_mode_prof_by_players["bens_t"] = bens_t
        dico_mode_prof_by_players["csts_t"] = csts_t
        dico_mode_prof_by_players["Perf_t"] = round(Perf_t,2)                   # utility of the game
        dico_mode_prof_by_players["b0_t"] = round(b0_t,2)
        dico_mode_prof_by_players["c0_t"] = round(c0_t,2)
        dico_mode_prof_by_players["Out_sg"] = round(Out_sg,2)
        dico_mode_prof_by_players["In_sg"] = round(In_sg,2)
        dico_mode_prof_by_players["pi_sg_plus_t"] = round(pi_sg_plus_t,2)
        dico_mode_prof_by_players["pi_sg_minus_t"] = round(pi_sg_minus_t,2)
        dico_mode_prof_by_players["pi_0_plus_t"] = round(pi_0_plus_t,2)
        dico_mode_prof_by_players["pi_0_minus_t"] = round(pi_0_minus_t,2)
        dico_mode_prof_by_players["mode_profile"] = mode_profile
        
        
        dico_Vis_Pref_t["Perf_t"] = Perf_t
        dico_Vis_Pref_t["b0"] = b0_t
        dico_Vis_Pref_t["c0"] = c0_t
            
        dico_profs_Vis_Perf_t[mode_profile] = dico_Vis_Pref_t
        
        cpt_profs += 1
            
    return dico_profs_Vis_Perf_t, cpt_profs, dico_mode_prof_by_players
# ________       genearte mode profile and its Perf_ts   ---> fin        ______


# _____       main function of NASH EQUILIBRIUM ONE ALGO  ---> debut      _____
def nash_balanced_player_game_ONE_ALGO(arr_pl_M_T_vars_init, algo_name,
                                    pi_hp_plus=0.02, 
                                    pi_hp_minus=0.33,
                                    a=1, b=1,
                                    gamma_version=1,
                                    path_to_save="tests", 
                                    name_dir="tests", 
                                    date_hhmm="DDMM_HHMM",
                                    manual_debug=False, 
                                    criteria_bf="Perf_t", dbg=False):
    
    print("\n \n game: {}, pi_hp_minus ={} ---> debut \n"\
          .format(pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,))                                # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_T = np.empty(shape=(t_periods,))                                  # shape (T_PERIODS,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_0_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    B_is_M_T = np.empty(shape=(m_players, t_periods))                           # shape (M_PLAYERS, )
    B_is_M_T.fill(np.nan)
    C_is_M_T = np.empty(shape=(m_players, t_periods))                           # shape (M_PLAYERS, )
    C_is_M_T.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,))                                      # shape (T_PERIODS,)
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods))                           # shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    CC_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    CC_is_M.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    BB_is_M.fill(np.nan)
    EB_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    EB_is_M.fill(np.nan)
    CC_is_M_T = np.empty(shape=(m_players, t_periods))                          # shape (M_PLAYERS, )
    CC_is_M_T.fill(np.nan)
    BB_is_M_T = np.empty(shape=(m_players, t_periods))                          # shape (M_PLAYERS, )
    BB_is_M_T.fill(np.nan)
    EB_is_M_T = np.empty(shape=(m_players, t_periods))                          # shape (M_PLAYERS, )
    EB_is_M_T.fill(np.nan)
    pi_hp_minus_T = np.empty(shape=(t_periods,))
    pi_hp_plus_T = np.empty(shape=(t_periods,))
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    dico_id_players = {"players":[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i) 
                                  for num_pl_i in range(0, m_players)]}
    df_nash = pd.DataFrame.from_dict(dico_id_players)
    
    # ____      game beginning for all t_period ---> debut      _____
    
    pi_sg_plus_t0_minus_1, pi_sg_minus_t0_minus_1 = None, None
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None, None
    pi_sg_plus_t, pi_sg_minus_t = None, None
    pi_hp_plus_t, pi_hp_minus_t = None, None
    
    dico_profils_NH, dico_bad_profils_NH, dico_best_profils_NH = dict(), dict(), dict()
    dico_profils_NH["profils"] = []; dico_profils_NH["Perfs"] = [];
    dico_bad_profils_NH["profils"] = []; dico_bad_profils_NH["Perfs"] = []; 
    dico_best_profils_NH["profils"] = []; dico_best_profils_NH["Perfs"] = []; 
    
    dico_modes_profs_by_players_t = dict() 
    for t in range(0, t_periods):
        print("----- t = {} , free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
        pi_hp_plus_t, pi_hp_minus_t = None, None
        pi_0_plus_t, pi_0_minus_t = None, None
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
            pi_hp_plus_t, pi_hp_minus_t = pi_hp_plus, pi_hp_minus
        else:
            q_t_minus, q_t_plus = fct_aux.compute_upper_bound_quantity_energy(
                                    arr_pl_M_T_vars_modif, t)
            phi_hp_minus_t = fct_aux.compute_cost_energy_bought_by_SG_2_HP(
                                pi_hp_minus=pi_hp_minus, 
                                quantity=q_t_minus,
                                b=b)
            phi_hp_plus_t = fct_aux.compute_benefit_energy_sold_by_SG_2_HP(
                                pi_hp_plus=pi_hp_plus, 
                                quantity=q_t_plus,
                                a=a)
            pi_hp_minus_t = round(phi_hp_minus_t/q_t_minus, fct_aux.N_DECIMALS) \
                            if q_t_minus != 0 \
                            else 0
            pi_hp_plus_t = round(phi_hp_plus_t/q_t_plus, fct_aux.N_DECIMALS) \
                            if q_t_plus != 0 \
                            else 0
            if t == 0:
                pi_sg_plus_t0_minus_1 = pi_hp_plus_t - 1
                pi_sg_minus_t0_minus_1 = pi_hp_minus_t - 1
            pi_sg_plus_t_minus_1 = pi_sg_plus_t0_minus_1 if t == 0 \
                                                         else pi_sg_plus_t
            pi_sg_minus_t_minus_1 = pi_sg_minus_t0_minus_1 if t == 0 \
                                                            else pi_sg_minus_t
            
            print("q_t-={}, phi_hp-={}, pi_hp-={}, pi_sg-_t-1={}, ".format(q_t_minus, phi_hp_minus_t, pi_hp_minus_t, pi_sg_minus_t_minus_1))
            print("q_t+={}, phi_hp+={}, pi_hp+={}, pi_sg+_t-1={}".format(q_t_plus, phi_hp_plus_t, pi_hp_plus_t, pi_sg_plus_t_minus_1))
            
            pi_0_plus_t = round(pi_sg_minus_t_minus_1*pi_hp_plus_t/pi_hp_minus_t, 
                                fct_aux.N_DECIMALS) \
                            if t > 0 \
                            else fct_aux.PI_0_PLUS_INIT #4
                                
            pi_0_minus_t = pi_sg_minus_t_minus_1 \
                            if t > 0 \
                            else fct_aux.PI_0_MINUS_INIT #3
            print("t={}, pi_0_plus_t={}, pi_0_minus_t={}".format(t, pi_0_plus_t, pi_0_minus_t))
               
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        pi_hp_plus_T[t] = pi_hp_plus_t
        pi_hp_minus_T[t] = pi_hp_minus_t
        pi_sg_plus_T[t] = pi_sg_plus_t_minus_1
        pi_sg_minus_T[t] = pi_sg_minus_t_minus_1
               
        arr_pl_M_T_vars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus_t=pi_hp_plus_t, pi_hp_minus_t=pi_hp_minus_t,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)
        print('gamma_is={}'.format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]))
        
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))
        
        # balanced player game at instant t    
        dico_profs_Vis_Perf_t = dict()
        mode_profiles = it.product(*possibles_modes)
        dico_profs_Vis_Perf_t, \
        cpt_profs, \
        dico_mode_prof_by_players = generer_mode_profiles(
                                        mode_profiles,
                                        arr_pl_M_T_vars_modif.copy(),
                                        t,
                                        pi_hp_plus, pi_hp_minus, 
                                        a, b,
                                        pi_0_plus_t, pi_0_minus_t,
                                        random_mode=False,
                                        manual_debug=manual_debug, dbg=False)
        
        # detection of NASH profils
        nash_profils = list();
        dico_nash_profils = list()
        nash_profils, dico_nash_profils \
            = detect_nash_balancing_profil(dico_profs_Vis_Perf_t,
                                           arr_pl_M_T_vars_modif, 
                                           t)
            
        ####### ___
        dico_profils_NH["nb_profils"] = len(nash_profils)
        for key, dico in dico_nash_profils.items():
            dico_profils_NH["profils"].append(dico["mode_profil"])
            dico_profils_NH["Perfs"].append(dico["Perf_t"])
        ####### ___    
        
        
        
        # delete all occurences of the profiles 
        print("----> avant supp doublons nash_profils={}".format(len(nash_profils)))
        nash_profils = set(nash_profils)
        print("----> apres supp doublons nash_profils={}".format(len(nash_profils)))
        
        # create dico of nash profils with key is Pref_t and value is profil
        dico_Perft_nashProfil = dict()
        for nash_mode_profil in nash_profils:
            Perf_t = dico_profs_Vis_Perf_t[nash_mode_profil]["Perf_t"]
            if Perf_t in dico_Perft_nashProfil:
                dico_Perft_nashProfil[Perf_t].append(nash_mode_profil)
            else:
                dico_Perft_nashProfil[Perf_t] = [nash_mode_profil]
        
        # min, max, mean of Perf_t
        best_key_Perf_t = None
        if algo_name == fct_aux.ALGO_NAMES_NASH[0]:                             # BEST-NASH
            best_key_Perf_t = max(dico_Perft_nashProfil.keys())
            best_NH_mode_profiles = dico_Perft_nashProfil[best_key_Perf_t]
            dico_best_profils_NH["nb_best_profils"] = len(best_NH_mode_profiles)
            for best_NH_mode_profile in best_NH_mode_profiles:
                dico_best_profils_NH["profils"].append(best_NH_mode_profile)
                dico_best_profils_NH["Perfs"].append(best_key_Perf_t)
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                           # BAD-NASH
            best_key_Perf_t = min(dico_Perft_nashProfil.keys())
            bad_NH_mode_profiles = dico_Perft_nashProfil[best_key_Perf_t]
            dico_bad_profils_NH["nb_bad_profils"] = len(bad_NH_mode_profiles)
            for bad_NH_mode_profile in bad_NH_mode_profiles:
                dico_bad_profils_NH["profils"].append(bad_NH_mode_profile)
                dico_bad_profils_NH["Perfs"].append(best_key_Perf_t)
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                           # MIDDLE-NASH
            mean_key_Perf_t  = np.mean(list(dico_Perft_nashProfil.keys()))
            if mean_key_Perf_t in dico_Perft_nashProfil.keys():
                best_key_Perf_t = mean_key_Perf_t
            else:
                sorted_keys = sorted(dico_Perft_nashProfil.keys())
                boolean = True; i_key = 1
                while boolean:
                    if sorted_keys[i_key] <= mean_key_Perf_t:
                        i_key += 1
                    else:
                        boolean = False; i_key -= 1
                best_key_Perf_t = sorted_keys[i_key]
                
        # find the best, bad, middle key in dico_Perft_nashProfil and 
        # the best, bad, middle nash_mode_profile
        best_nash_mode_profiles = dico_Perft_nashProfil[best_key_Perf_t]
        best_nash_mode_profile = None
        if len(best_nash_mode_profiles) == 1:
            best_nash_mode_profile = best_nash_mode_profiles[0]
        else:
            rd = np.random.randint(0, len(best_nash_mode_profiles))
            best_nash_mode_profile = best_nash_mode_profiles[rd]
        
        print("** Running at t={}: numbers of -> cpt_profils={}, nash_profils={}, {}_nash_mode_profiles={}, nbre_cle_dico_Perft_nashProf={}"\
              .format(t, cpt_profs, 
                len(nash_profils),
                algo_name.split("-")[0], len(best_nash_mode_profiles),
                list(dico_Perft_nashProfil.keys())      
                      ))
        print("{}_key_Perf_t={}, {}_nash_mode_profile={}".format(
                algo_name.split("-")[0], best_key_Perf_t, 
                algo_name.split("-")[0], 
                best_nash_mode_profile))
        
        arr_pl_M_T_vars_nash_mode_prof, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t_nash_mode_prof \
            = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                arr_pl_M_T_vars_modif.copy(),
                best_nash_mode_profile, t,
                pi_hp_plus, pi_hp_minus, 
                a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                random_mode=False,
                manual_debug=manual_debug, dbg=dbg
                )
            
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
        
        # checkout NASH equilibrium
        bens_csts_M_t = bens_t - csts_t
        df_nash_t = None        
        df_nash_t = fct_aux.checkout_nash_4_profils_by_periods(
                        arr_pl_M_T_vars_nash_mode_prof.copy(),
                        arr_pl_M_T_vars_init,
                        pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                        a=a, b=b,
                        pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                        bens_csts_M_t=bens_csts_M_t,
                        t=t,
                        manual_debug=manual_debug)
        df_nash = pd.merge(df_nash, df_nash_t, 
                                on='players', 
                                how='outer')
        
        #_______     save arr_M_t_vars at t in dataframe : debut    _______
        # df_arr_M_t_vars_modif \
        #     = pd.DataFrame(arr_pl_M_T_vars_modif[:,t,:], 
        #                     columns=fct_aux.AUTOMATE_INDEX_ATTRS.keys(),
        #                     index=dico_id_players["players"])
        # path_to_save_M_t_vars_modif = path_to_save
        # msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
        # if "simu_DDMM_HHMM" in path_to_save:
        #     path_to_save_M_t_vars_modif \
        #         = os.path.join(name_dir, "simu_"+date_hhmm,
        #                        msg, algo_name, "intermediate_t"
        #                        )
        # Path(path_to_save_M_t_vars_modif).mkdir(parents=True, 
        #                                         exist_ok=True)
            
        # path_2_xls_df_arr_M_t_vars_modif \
        #     = os.path.join(
        #         path_to_save_M_t_vars_modif,
        #           "arr_M_T_vars_{}.xlsx".format(algo_name)
        #         )
        # if not os.path.isfile(path_2_xls_df_arr_M_t_vars_modif):
        #     df_arr_M_t_vars_modif.to_excel(
        #         path_2_xls_df_arr_M_t_vars_modif,
        #         sheet_name="t{}".format(t),
        #         index=True)
        # else:
        #     book = load_workbook(filename=path_2_xls_df_arr_M_t_vars_modif)
        #     with pd.ExcelWriter(path_2_xls_df_arr_M_t_vars_modif, 
        #                         engine='openpyxl') as writer:
        #         writer.book = book
        #         writer.sheets = dict((ws.title, ws) for ws in book.worksheets)    
            
        #         ## Your dataframe to append. 
        #         df_arr_M_t_vars_modif.to_excel(writer, "t{}".format(t))  
            
        #         writer.save() 
        #_______     save arr_M_t_vars at t in dataframe : fin     _______
    
        #___________ update saving variables : debut ______________________
        arr_pl_M_T_vars_modif[:,t,:] = arr_pl_M_T_vars_nash_mode_prof[:,t,:].copy()        

        b0_ts_T, c0_ts_T, \
        BENs_M_T, CSTs_M_T, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        df_nash \
            = autoBfGameModel4T.update_saving_variables(t, 
                b0_ts_T, b0_t,
                c0_ts_T, c0_t,
                BENs_M_T, bens_t,
                CSTs_M_T, csts_t,
                pi_sg_plus_T, pi_sg_plus_t,
                pi_sg_minus_T, pi_sg_minus_t,
                pi_0_plus_T, pi_0_plus_t,
                pi_0_minus_T, pi_0_minus_t,
                df_nash, df_nash
                )
        dico_modes_profs_by_players_t[t] = dico_mode_prof_by_players
        #___________ update saving variables : fin   ______________________
        print("Sis = {}".format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))

        print("----- t={} After running free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
    
    # __________        compute prices variables         ____________________
    B_is_M, C_is_M, BB_is_M, CC_is_M, EB_is_M, \
    B_is_MT_cum, C_is_MT_cum, \
    B_is_M_T, C_is_M_T, BB_is_M_T, CC_is_M_T, EB_is_M_T \
        = fct_aux.compute_prices_B_C_BB_CC_EB_DET(
                arr_pl_M_T_vars_modif=arr_pl_M_T_vars_modif, 
                pi_sg_minus_T=pi_sg_minus_T, pi_sg_plus_T=pi_sg_plus_T, 
                pi_0_minus_T=pi_0_minus_T, pi_0_plus_T=pi_0_plus_T,
                b0_s_T=b0_ts_T, c0_s_T=c0_ts_T)
        
    dico_EB_R_EBsetA1B1_EBsetB2C = {"EB_setA1B1":[np.nan],"EB_setB2C":[np.nan], 
                                    "ER":[np.nan], "VR":[np.nan]}
    # __________        compute prices variables         ____________________
    
    #_______      save computed variables locally from algo_name     __________
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    
    print("path_to_save={}".format(path_to_save))
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    # df_nash.to_excel(os.path.join(
    #             *[path_to_save,
    #               "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
    #             index=False)
    df_nash.to_csv(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.csv".format(algo_name)]), 
                index=False)

    fct_aux.save_variables(
            path_to_save=path_to_save, 
            arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif, 
            b0_s_T_K=b0_ts_T, c0_s_T_K=c0_ts_T, 
            B_is_M=B_is_M, C_is_M=C_is_M, B_is_M_T=B_is_M_T, C_is_M_T=C_is_M_T,
            BENs_M_T_K=BENs_M_T, CSTs_M_T_K=CSTs_M_T, 
            BB_is_M=BB_is_M, CC_is_M=CC_is_M, EB_is_M=EB_is_M, 
            BB_is_M_T=BB_is_M_T, CC_is_M_T=CC_is_M_T, EB_is_M_T=EB_is_M_T,
            dico_EB_R_EBsetA1B1_EBsetB2C=dico_EB_R_EBsetA1B1_EBsetB2C,
            pi_sg_minus_T_K=pi_sg_minus_T, pi_sg_plus_T_K=pi_sg_plus_T, 
            pi_0_minus_T_K=pi_0_minus_T, pi_0_plus_T_K=pi_0_plus_T,
            pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T, 
            dico_stats_res=dico_modes_profs_by_players_t, 
            algo=algo_name, 
            dico_best_steps=dict())
    autoBfGameModel4T.turn_dico_stats_res_into_df_BF(
          dico_modes_profs_players_algo = dico_modes_profs_by_players_t, 
          path_to_save = path_to_save, 
          t_periods = t_periods, 
          manual_debug = manual_debug, 
          algo_name = algo_name)
    
    return arr_pl_M_T_vars_modif, dico_profils_NH, \
            dico_best_profils_NH, dico_bad_profils_NH
# _____       main function of NASH EQUILIBRIUM ONE ALGO  --->  fin       _____


###############################################################################
#                   definition  des unittests
#
###############################################################################
def OLD_test_NASH_balanced_player_game_Pi_Ci_one_period():
    
    fct_aux.N_DECIMALS = 8
    a=1; b=1;
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 20 #0.33 #[15, 5]
    
    manual_debug = False #True
    gamma_version = 1 #1,2
    debug = False
    criteria_bf = "Perf_t"
    used_instances = False#True #False#True
    
    setA_m_players = 15; setB_m_players = 10; setC_m_players = 10
    setA_m_players, setB_m_players, setC_m_players = 8, 3, 3
    setA_m_players, setB_m_players, setC_m_players = 6, 2, 2
    setA_m_players, setB_m_players, setC_m_players = 1, 1, 1
    t_periods = 1 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    
    
    prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario = [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
    scenario_name = "scenarioABC"
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_one_period(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, 
                        scenario,
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl_one_period(arr_pl_M_T_vars_init)
    
    algo_name = fct_aux.ALGO_NAMES_NASH[1]
    name_simu = algo_name+"_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    arr_pl_M_T_vars, dico_profils_NH, \
    dico_best_profils_NH, dico_bad_profils_NH \
        = nash_balanced_player_game_ONE_ALGO(
                            arr_pl_M_T_vars_init.copy(),
                            algo_name=algo_name,
                            pi_hp_plus=pi_hp_plus, 
                            pi_hp_minus=pi_hp_minus,
                            gamma_version=gamma_version,
                            path_to_save=path_to_save, 
                            name_dir="tests", 
                            date_hhmm="DDMM_HHMM",
                            manual_debug=manual_debug, 
                            criteria_bf = criteria_bf,
                            dbg=debug)
    
    return arr_pl_M_T_vars, dico_profils_NH, dico_best_profils_NH, \
            dico_bad_profils_NH 
    
def test_NASH_balanced_player_game_Pi_Ci_one_period():
    
    fct_aux.N_DECIMALS = 8
    a, b = 1, 1
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 20 #0.33 #[15, 5]
    
    manual_debug = False
    gamma_version = -2  #-2,-1,1,2,3,4,5
    debug = False
    criteria_bf = "Perf_t"
    used_instances = True #False #True
    
    t_periods = 1
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    
    # _____                     scenarios --> debut                 __________
    scenario_name = "scenarioOnePeriod"      
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
    arr_pl_M_T_vars_init \
        = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc23(
            setA_m_players=setA_m_players, 
            setB_m_players=setB_m_players, 
            setC_m_players=setC_m_players, 
            t_periods=t_periods, 
            scenario=None,
            scenario_name=scenario_name,
            path_to_arr_pl_M_T=path_to_arr_pl_M_T, 
            used_instances=used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl_one_period_doc23(arr_pl_M_T_vars_init)
    # _____                     scenarios --> fin                   __________
    
    global MOD
    m_players = setA_m_players + setB_m_players + setC_m_players
    MOD = int(0.10*pow(2, m_players)) \
            if pow(2, m_players) < 65000 \
            else int(0.020*pow(2, m_players))    
    
    algo_name = fct_aux.ALGO_NAMES_NASH[np.random.randint(low=0, high=3)]
    name_simu = algo_name+"_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    arr_pl_M_T_vars, dico_profils_bf, \
    dico_best_profils_bf, dico_bad_profils_bf \
        = nash_balanced_player_game_ONE_ALGO(
                        arr_pl_M_T_vars_init.copy(),
                        algo_name=algo_name,
                        pi_hp_plus=pi_hp_plus, 
                        pi_hp_minus=pi_hp_minus,
                        a=a, b=b,
                        gamma_version = gamma_version,
                        path_to_save=path_to_save, 
                        name_dir="tests", 
                        date_hhmm="DDMM_HHMM",
                        manual_debug=manual_debug, 
                        criteria_bf=criteria_bf, dbg=debug)
    
    return arr_pl_M_T_vars, dico_profils_bf, \
            dico_best_profils_bf, dico_bad_profils_bf
###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    # arr_pl_M_T_vars = test_NASH_balanced_player_game()
    arr_pl_M_T_vars, dico_profils_NH, \
    dico_best_profils_NH, dico_bad_profils_NH \
        = test_NASH_balanced_player_game_Pi_Ci_one_period()
    
    print("runtime = {}".format(time.time() - ti))