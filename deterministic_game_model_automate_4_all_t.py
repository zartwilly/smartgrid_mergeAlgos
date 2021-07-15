# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:15:32 2021

@author: jwehounou
"""

import os
import time

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from pathlib import Path
from datetime import datetime

###############################################################################
#                   definition  des fonctions annexes
#
###############################################################################

# _______        balanced players at t and k --> debut          ______________
def balanced_player_game_4_random_mode(arr_pl_M_T_vars_modif, t, 
                                       algo_name, 
                                       used_storage, 
                                       manual_debug, dbg):
    
    dico_gamma_players_t = dict()
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']] 
        Si_max = arr_pl_M_T_vars_modif[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i, state_i = 0, 0, 0, ""
        state_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                  fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # get mode_i
        mode_i = None
        if algo_name == fct_aux.ALGO_NAMES_DET[0]:                                    # Systematic-DETERMINIST
            if state_i == fct_aux.STATES[0]:
                mode_i = fct_aux.STATE1_STRATS[0]                               # CONS+, Deficit
            elif state_i == fct_aux.STATES[1]:
                mode_i = fct_aux.STATE2_STRATS[0]                               # DIS, Self
            elif state_i == fct_aux.STATES[2]:
                mode_i = fct_aux.STATE3_STRATS[0]                               # DIS, Surplus
            pl_i.set_mode_i(mode_i)
        else:                                                                   # Selfish-DETERMINIST
            # t in [1,num_periods]
            Pi_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, 
                                          t+1, 
                                          fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else 0
            Ci_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, 
                                     t+1, 
                                     fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1 < t_periods \
                            else 0
            # Si_t_minus_1_minus = arr_pl_M_T_vars_modif[num_pl_i, 
            #                         t-1, 
            #                         fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] \
            #                     if t-1 > 0 \
            #                     else 0
            Si_t_minus = arr_pl_M_T_vars_modif[num_pl_i, 
                                    t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]]
            # Si_t_minus_1_plus = arr_pl_M_T_vars_modif[num_pl_i, 
            #                          t-1, 
            #                          fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] \
            #                     if t-1 > 0 \
            #                     else 0
            
            if used_storage:
                if state_i == fct_aux.STATES[0] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE1_STRATS[0]                          # CONS+, Deficit
                elif state_i == fct_aux.STATES[0] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE1_STRATS[1]                          # CONS-, Deficit
                elif state_i == fct_aux.STATES[1] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE2_STRATS[1]                          # CONS-, Self
                elif state_i == fct_aux.STATES[1] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE2_STRATS[0]                          # DIS, Self
                elif state_i == fct_aux.STATES[2] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE3_STRATS[0]                          # DIS, Surplus
                elif state_i == fct_aux.STATES[2] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE3_STRATS[1]                          # PROD, Surplus
            else:
                if state_i == fct_aux.STATES[0]:
                    mode_i = fct_aux.STATE1_STRATS[1]           # CONS-, Deficit
                elif state_i == fct_aux.STATES[1]:
                    mode_i = fct_aux.STATE2_STRATS[0]           # DIS, Self
                elif state_i == fct_aux.STATES[2]:
                    mode_i = fct_aux.STATE3_STRATS[1]           # PROD, Surplus
                    
            pl_i.set_mode_i(mode_i)
                
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
    
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_modif
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), 
                ("mode_i", pl_i.get_mode_i()), ("state_i", pl_i.get_state_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_M_T_vars_modif[num_pl_i, t, 
                                  fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars_modif, dico_gamma_players_t


def balanced_player_game_t(arr_pl_M_T_vars_modif, t, 
                            pi_hp_plus, pi_hp_minus,
                            a, b,
                            pi_0_plus_t, pi_0_minus_t,
                            algo_name, used_storage,
                            manual_debug, dbg):
    # find mode, prod, cons, r_i
    arr_pl_M_T_vars_modif, dico_gamma_players_t \
        = balanced_player_game_4_random_mode(
            arr_pl_M_T_vars_modif.copy(), t, 
            algo_name, 
            used_storage, 
            manual_debug, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t, \
        = fct_aux.compute_prices_inside_SG_4_notLearnAlgo(
            arr_pl_M_T_vars_modif, t,
            pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
            a=a, b=b,
            pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
            manual_debug=manual_debug, dbg=dbg)
        
    return arr_pl_M_T_vars_modif, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamma_players_t
# _______        balanced players at t and k --> fin            ______________

# __________       main function of DETERMINIST   ---> debut      ____________
def determinist_balanced_player_game(arr_pl_M_T_vars_init,
                                     pi_hp_plus=0.2, 
                                     pi_hp_minus=0.33,
                                     a=1, b=1,
                                     gamma_version=1,
                                     algo_name=fct_aux.ALGO_NAMES_DET[0],
                                     used_storage=False,
                                     path_to_save="tests", 
                                     manual_debug=False, dbg=False):
    
    """
    create a game for balancing all players at all periods of time T_PERIODS = [0..T-1]

    Parameters
    ----------
    arr_pl_M_T: array of shape (M_PLAYERS, T_PERIODS, len(AUTOMATE_INDEX_ATTRS))
        DESCRIPTION.
    pi_hp_plus : float, optional
        DESCRIPTION. The default is 0.10.
        the price of exported energy from SG to HP
    pi_hp_minus : float, optional
        DESCRIPTION. The default is 0.15.
        the price of imported energy from HP to SG
    random_determinist: boolean, optional
        DESCRIPTION. The default is False
        decide if the mode of player a_i is randomly chosen (True) or 
        deterministly chosen (False) 
    path_to_save : String, optional
        DESCRIPTION. The default is "tests".
        name of directory for saving variables of players
    dbg : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    
    """
    print("determinist game: pi_hp_plus={}, pi_hp_minus ={} ---> debut \n"\
          .format( pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T = np.empty(shape=(t_periods,)); pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,)); pi_sg_plus_T.fill(np.nan)
    pi_0_plus_T = np.empty(shape=(t_periods,)); pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,)); pi_0_minus_T.fill(np.nan)
    pi_hp_plus_T = np.empty(shape=(t_periods, )); pi_hp_plus_T.fill(np.nan)
    pi_hp_minus_T = np.empty(shape=(t_periods, )); pi_hp_minus_T.fill(np.nan)
    b0_s_T = np.empty(shape=(t_periods,)); b0_s_T.fill(np.nan)
    c0_s_T = np.empty(shape=(t_periods,)); c0_s_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))    
    prod_M_T = np.empty(shape=(m_players, t_periods)); prod_M_T.fill(np.nan)
    cons_M_T = np.empty(shape=(m_players, t_periods)); cons_M_T.fill(np.nan)
    B_is_M_T = np.empty(shape=(m_players, t_periods)); B_is_M_T.fill(np.nan)
    C_is_M_T = np.empty(shape=(m_players, t_periods)); C_is_M_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players, )); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players, )); C_is_M.fill(np.nan)
    BB_is_M_T = np.empty(shape=(m_players, t_periods)); BB_is_M_T.fill(np.nan)
    CC_is_M_T = np.empty(shape=(m_players, t_periods)); CC_is_M_T.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players, )); BB_is_M.fill(np.nan)
    CC_is_M = np.empty(shape=(m_players, )); CC_is_M.fill(np.nan)
    RU_is_M = np.empty(shape=(m_players, )); RU_is_M.fill(np.nan)
    
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____          run balanced sg for all t_periods : debut         ________
    dico_stats_res = dict()
    dico_mode_prof_by_players_T = dict()
    
    dico_id_players = {"players":[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i) 
                                  for num_pl_i in range(0, m_players)]}
    df_nash = pd.DataFrame.from_dict(dico_id_players)
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    
    pi_sg_plus_t0_minus_1, pi_sg_minus_t0_minus_1 = None, None
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None, None
    pi_sg_plus_t, pi_sg_minus_t = None, None
    pi_hp_plus_t, pi_hp_minus_t = None, None
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
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
            
            pi_0_plus_t = None
            if pi_hp_minus_t > 0:
                pi_0_plus_t = round(pi_sg_minus_t_minus_1*pi_hp_plus_t/pi_hp_minus_t, 
                                    fct_aux.N_DECIMALS)
            else:
                pi_0_plus_t = 0
                            
            pi_0_plus_t =  pi_0_plus_t if t > 0 else fct_aux.PI_0_PLUS_INIT #4
                                
            pi_0_minus_t = pi_sg_minus_t_minus_1 \
                            if t > 0 \
                            else fct_aux.PI_0_MINUS_INIT #3
            print("t={}, pi_0_plus_t={}, pi_0_minus_t={}".format(t, pi_0_plus_t, pi_0_minus_t))
               
        if t == 0:
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
            )) if dbg else None
        else:
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
                t-1,
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
                )) if dbg else None
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
                t,
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
                )) if dbg else None
            
        arr_pl_M_T_vars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus_t=pi_hp_plus_t, pi_hp_minus_t=pi_hp_minus_t,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)
        
        print("after compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
            )) if dbg else None
            
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        pi_hp_plus_T[t] = pi_hp_plus_t
        pi_hp_minus_T[t] = pi_hp_minus_t
        pi_sg_plus_T[t] = pi_sg_plus_t_minus_1
        pi_sg_minus_T[t] = pi_sg_minus_t_minus_1
        
        # balanced player game at instant t
        dico_gamme_t = dict()
        arr_pl_M_T_vars_modif, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t \
            = balanced_player_game_t(
                arr_pl_M_T_vars_modif.copy(), t, 
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                algo_name=algo_name, used_storage=used_storage,
                manual_debug=manual_debug, dbg=dbg)
        Sis = arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]].copy()
        arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]] = Sis
        print("after Sis update 4 t={}, Sis={}, modes={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]
            )) if dbg else None
        dico_stats_res[t] = dico_gamme_t
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
        
        # b0_ts, c0_ts of shape (T_PERIODS,)
        b0_s_T[t] = b0_t
        c0_s_T[t] = c0_t
        
        # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        # compute Perf_t
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_modif, 
                                t)
        
        bens_csts_M_t = bens_t - csts_t
        Perf_t = np.sum(bens_csts_M_t, axis=0)
        dico_players = dict()
        for num_pl_i in range(0, m_players):
            Vi = bens_csts_M_t[num_pl_i]
            
            dico_vars = dict()
            dico_vars["Vi"] = round(Vi, 2)
            dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
            dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
            variables = ["set", "state_i", "mode_i", "Pi", "Ci", "Si_max", 
                         "Si_old", "Si", "prod_i", "cons_i", "r_i", 
                         "Si_minus", "Si_plus", "gamma_i"]
            for variable in variables:
                dico_vars[variable] = arr_pl_M_T_vars_modif[
                                        num_pl_i, t, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                
            dico_players[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)] = dico_vars
        
        dico_players["Perf_t"] = round(Perf_t, 2)
        dico_players["b0_t"] = round(b0_t, 2)
        dico_players["c0_t"] = round(c0_t, 2)
        dico_players["Out_sg"] = round(Out_sg, 2)
        dico_players["In_sg"] = round(In_sg, 2)
        dico_players["pi_sg_plus_t"] = round(pi_sg_plus_t, 2)
        dico_players["pi_sg_minus_t"] = round(pi_sg_minus_t, 2)
        dico_players["pi_0_plus_t"] = round(pi_0_plus_t, 2)
        dico_players["pi_0_minus_t"] = round(pi_0_minus_t, 2)
        
        dico_mode_prof_by_players_T["t_"+str(t)] = dico_players
        
        ## checkout NASH equilibrium
        df_nash_t = None
        df_nash_t = fct_aux.checkout_nash_4_profils_by_periods(
                        arr_pl_M_T_vars_modif.copy(),
                        arr_pl_M_T_vars_init,
                        pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                        a=a, b=b,
                        pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                        bens_csts_M_t=bens_csts_M_t,
                        t=t,
                        manual_debug=manual_debug)
        df_nash = pd.merge(df_nash, df_nash_t, on='players', how='outer')
    
    # ____          run balanced sg for all t_periods : fin           ________
    
    # __________        compute prices variables         ____________________
    B_is_M, C_is_M, BB_is_M, CC_is_M, EB_is_M, \
    B_is_M_T, C_is_M_T, BB_is_M_T, CC_is_M_T, EB_is_M_T, \
    B_is_MT, C_is_MT \
        = fct_aux.compute_prices_B_C_BB_CC_EB_DET(
                arr_pl_M_T_vars_modif=arr_pl_M_T_vars_modif, 
                pi_sg_minus_T=pi_sg_minus_T, pi_sg_plus_T=pi_sg_plus_T, 
                pi_0_minus_T=pi_0_minus_T, pi_0_plus_T=pi_0_plus_T,
                b0_s_T=b0_s_T, c0_s_T=c0_s_T)
    
    VR = np.sum(np.sum( B_is_MT - C_is_MT, axis=0), axis=0)
    ER = np.sum(EB_is_M, axis=0)
    
    dico_EB_R_EBsetA1B1_EBsetB2C = {"EB_setA1B1":[np.nan],"EB_setB2C":[np.nan], 
                                    "ER":[np.nan], "VR":[np.nan]}
    set_Mplayers = np.unique(arr_pl_M_T_vars_modif[:, 0, fct_aux.AUTOMATE_INDEX_ATTRS['set']]).tolist()
    if set(set_Mplayers).intersection(fct_aux.SET_AB1B2C) == set(fct_aux.SET_AB1B2C):
        setA1B1, setB2C = list(), list()
        setA1B1 \
            = np.argwhere(
                (arr_pl_M_T_vars_modif[:, 0, fct_aux.AUTOMATE_INDEX_ATTRS['set']] == "setA") | 
                (arr_pl_M_T_vars_modif[:, 0, fct_aux.AUTOMATE_INDEX_ATTRS['set']] == "setB1")
                ).reshape(-1).tolist()
        setB2C \
            = np.argwhere(
                (arr_pl_M_T_vars_modif[:, 0, fct_aux.AUTOMATE_INDEX_ATTRS['set']] == "setB2" ) | 
                (arr_pl_M_T_vars_modif[:, 0, fct_aux.AUTOMATE_INDEX_ATTRS['set']] == "setC" )
                ).reshape(-1).tolist()
        EB_setA1B1_det = np.sum(EB_is_M[setA1B1], axis=0)
        EB_setB2C_det = np.sum(EB_is_M[setB2C], axis=0)
        dico_EB_R_EBsetA1B1_EBsetB2C = {"EB_setA1B1":[EB_setA1B1_det], 
                                         "EB_setB2C":[EB_setB2C_det], 
                                         "ER":[ER], "VR":[VR]}
    #__________      save computed variables locally      _____________________ 
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)
    
    if m_players<=22:
        fct_aux.save_variables(
            path_to_save=path_to_save, 
            arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif, 
            b0_s_T_K=b0_s_T, c0_s_T_K=c0_s_T, 
            B_is_M=B_is_M, C_is_M=C_is_M, B_is_M_T=B_is_M_T, C_is_M_T=C_is_M_T,
            BENs_M_T_K=BENs_M_T, CSTs_M_T_K=CSTs_M_T, 
            BB_is_M=BB_is_M, CC_is_M=CC_is_M, EB_is_M=EB_is_M, 
            BB_is_M_T=BB_is_M_T, CC_is_M_T=CC_is_M_T, EB_is_M_T=EB_is_M_T,
            dico_EB_R_EBsetA1B1_EBsetB2C=dico_EB_R_EBsetA1B1_EBsetB2C,
            pi_sg_minus_T_K=pi_sg_minus_T, pi_sg_plus_T_K=pi_sg_plus_T, 
            pi_0_minus_T_K=pi_0_minus_T, pi_0_plus_T_K=pi_0_plus_T,
            pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T, 
            dico_stats_res=dico_stats_res, 
            algo=algo_name, 
            dico_best_steps=dico_mode_prof_by_players_T)
    else:
        fct_aux.save_variables(
            path_to_save=path_to_save, 
            arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif, 
            b0_s_T_K=b0_s_T, c0_s_T_K=c0_s_T, 
            B_is_M=B_is_M, C_is_M=C_is_M, B_is_M_T=B_is_M_T, C_is_M_T=C_is_M_T,
            BENs_M_T_K=BENs_M_T, CSTs_M_T_K=CSTs_M_T, 
            BB_is_M=BB_is_M, CC_is_M=CC_is_M, EB_is_M=EB_is_M, 
            BB_is_M_T=BB_is_M_T, CC_is_M_T=CC_is_M_T, EB_is_M_T=EB_is_M_T,
            dico_EB_R_EBsetA1B1_EBsetB2C=dico_EB_R_EBsetA1B1_EBsetB2C,
            pi_sg_minus_T_K=pi_sg_minus_T, pi_sg_plus_T_K=pi_sg_plus_T, 
            pi_0_minus_T_K=pi_0_minus_T, pi_0_plus_T_K=pi_0_plus_T,
            pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T, 
            dico_stats_res=dico_stats_res, 
            algo=algo_name, 
            dico_best_steps=dico_mode_prof_by_players_T)
        
        
    # _____         checkout prices from computing variables: debut      _____7
    dbg=True
    if dbg:
        fct_aux.checkout_prices_B_C_BB_CC_EB_DET(
                arr_pl_M_T_vars_modif=arr_pl_M_T_vars_modif, 
                path_to_save=path_to_save)
    # _____         checkout prices from computing variables: fin        _____
        
    print("DETERMINIST GAME: pi_hp_plus={}, pi_hp_minus ={} ---> FIN \n"\
          .format( pi_hp_plus, pi_hp_minus))
    
    return arr_pl_M_T_vars_modif
    
    
# __________       main function of DETERMINIST   ---> fin        ____________

###############################################################################
#                   definition  des unittests
#
###############################################################################

def test_DETERMINIST_balanced_player_game_Pi_Ci_scenario123():
    a = 1; b = 1
    pi_hp_plus = 10 #0.2*pow(10,-3)
    pi_hp_minus = 20
    used_storage = True #False
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False
    t_periods = 4

    dbg = False #True
    manual_debug = False
    gamma_version = 5 #2 #1 #3: gamma_i_min #4: square_root # 5: proba of 
    
    scenarios_name = ["scenario1", "scenario2", "scenario3"]  
    scenario_name = scenarios_name[np.random.randint(low=0, high=len(scenarios_name))] 
    prob_scen, scenario = None, None
    if scenario_name == "scenario1":
        prob_scen = 0.6
        prob_A_A = prob_scen; prob_A_C = 1-prob_scen;
        prob_C_A = 1-prob_scen; prob_C_C = prob_scen;
        scenario = [(prob_A_A, prob_A_C), 
                    (prob_C_A, prob_C_C)]
        setA_m_players_1 = 10; setC_m_players_1 = 10; 
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC_doc23(
                            setA_m_players_1, setC_m_players_1, 
                            t_periods, 
                            scenario,
                            scenario_name,
                            path_to_arr_pl_M_T, used_instances)
        fct_aux.checkout_values_Pi_Ci_arr_pl_SETAC_doc23(arr_pl_M_T_vars_init, 
                                                         scenario_name)
        
    elif scenario_name in ["scenario2", "scenario3"]:
        prob_scen = 0.8 if scenario_name == "scenario2" else 0.5
        prob_A_A = 1-prob_scen; prob_A_B1 = prob_scen; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = prob_scen; prob_B1_B1 = 1-prob_scen; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 1-prob_scen; prob_B2_C = prob_scen;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = prob_scen; prob_C_C = 1-prob_scen 
        scenario = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        setA_m_players_23 = 10; setB1_m_players_23 = 3; 
        setB2_m_players_23 = 5; setC_m_players_23 = 8; 
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc23(
                        setA_m_players_23, setB1_m_players_23, 
                        setB2_m_players_23, setC_m_players_23, 
                        t_periods, 
                        scenario,
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
        fct_aux.checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc23(arr_pl_M_T_vars_init, 
                                                             scenario_name)
    rd = np.random.randint(low=0, high=2)
    algo_name = fct_aux.ALGO_DET[rd]
    name_simu = algo_name+"_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    arr_pl_M_T_vars = \
        determinist_balanced_player_game(
                                 arr_pl_M_T_vars_init.copy(),
                                 pi_hp_plus=pi_hp_plus, 
                                 pi_hp_minus=pi_hp_minus,
                                 a=a, b=b,
                                 gamma_version=gamma_version,
                                 algo_name=algo_name,
                                 used_storage=used_storage,
                                 path_to_save=path_to_save, 
                                 manual_debug=manual_debug, dbg=dbg)
        
    return arr_pl_M_T_vars
    
###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    # arr_pl_M_T_K_vars_modif \
    #     = test_DETERMINIST_balanced_player_game_Pi_Ci_NEW_AUTOMATE()
        
    arr_pl_M_T_K_vars_modif \
        = test_DETERMINIST_balanced_player_game_Pi_Ci_scenario123()
    
    print("runtime = {}".format(time.time() - ti))