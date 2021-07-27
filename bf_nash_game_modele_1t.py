#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 07:50:47 2021

@author: willy
"""

import os
import time
import psutil

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import force_brute_game_model_automate_4_all_t as bf_game
import itertools as it

from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook

global MOD
MOD = 65000 #int(0.10*pow(2, m_players))


###############################################################################
#                   definition  des fonctions annexes
#
###############################################################################

# __________            find possibles modes --> debut               _________
def possibles_modes_players_automate(arr_pl_M_T_vars, m_players, t):
    """
    generate the list of possible modes by the states of players

    Parameters
    ----------
    arr_pl_M_T_vars : TYPE, shape (m_players,)
        DESCRIPTION. The default is None.
        it means that t and k are the fixed values 
    t : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.possibles_modes_players_automate

    """

    possibles_modes = list()
    
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_M_T_vars[num_pl_i, t,
                                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] 
        
        # get mode_i
        if state_i == fct_aux.STATES[0]:                                        # state1 or Deficit
            possibles_modes.append(fct_aux.STATE1_STRATS)
        elif state_i == fct_aux.STATES[1]:                                      # state2 or Self
            possibles_modes.append(fct_aux.STATE2_STRATS)
        elif state_i == fct_aux.STATES[2]:                                      # state3 or surplus
            possibles_modes.append(fct_aux.STATE3_STRATS)
            # print("3: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
        
    return possibles_modes
# __________            find possibles modes --> debut               _________

# ________       balanced players 4 one modes_profile   ---> debut      ______
def balanced_player_game_4_mode_profil(arr_pl_Mtvars_mode_prof, 
                                       m_players, t, dbg):
    """
    attribute modes of all players and get players' variables as prod_i, 
    cons_i, r_i, gamma_i saved to  arr_pl_M_T_vars_mode_prof

    Parameters
    ----------
    arr_pl_Mtvars_mode_prof : shape (m_players, t, len(AUTOMATE_INDEX_ATTRS))
        DESCRIPTION.
    m_players : number of players
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_t_vars_mode_prof

    """
    
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Si']]
        Si_max = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                        fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                       fct_aux.AUTOMATE_INDEX_ATTRS['gamma_i']]
        state_i = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        mode_i = arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['mode_i']]
        
        prod_i, cons_i, r_i = 0, 0, 0
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        
        pl_i.set_R_i_old(Si_max-Si)
        pl_i.set_mode_i(mode_i)
        
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
        
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_modif
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", mode_i), 
                ("gamma_i", gamma_i),
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_Mtvars_mode_prof[num_pl_i, t,
                                    fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_Mtvars_mode_prof
# ________       balanced players 4 one modes_profile   --->   fin      ______

# ________      checkout nash for mode profile ---> debut        ______________
def checkout_nash_notLearnAlgo(arr_pl_Mtvars_mode_prof, 
                               mode_profile, bens_csts_t,
                               m_players, t, 
                               pi_hp_plus, pi_hp_minus, a, b,
                               pi_0_plus_t, pi_0_minus_t, manual_debug, dbg):
    """
    check out if the mode profile is a nash equilibrium
    """
    cpt_STABLE_players = 0
    bool_NASH_modeprofile = True
    while bool_NASH_modeprofile and cpt_STABLE_players < m_players:
        num_pl_i = cpt_STABLE_players
        Vi = bens_csts_t[num_pl_i]
        mode_profile_bar = None
        mode_profile_bar = list(mode_profile).copy()
        state_i = arr_pl_Mtvars_mode_prof[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
        mode_i = arr_pl_Mtvars_mode_prof[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]
        mode_i_bar = None
        mode_i_bar = fct_aux.find_out_opposite_mode(state_i, mode_i)
        mode_profile_bar[num_pl_i] = mode_i_bar
        
        arr_pl_Mtvars_mode_prof_bar, \
        b0_t_bar, c0_t_bar, \
        bens_t_bar, csts_t_bar, \
        pi_sg_plus_t_bar, pi_sg_minus_t_bar, \
        dico_gamme_t \
            = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                    arr_pl_M_T_vars_modif=arr_pl_Mtvars_mode_prof,
                    mode_profile=mode_profile_bar, t=t,
                    pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                    a=a, b=b,
                    pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                    random_mode=False,
                    manual_debug=manual_debug, dbg=dbg
                    )
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_Mtvars_mode_prof_bar, t)
            
        # compute Perf_t_bar
        bens_csts_t_bar = bens_t_bar - csts_t_bar
        
        Vi_bar = bens_csts_t_bar[num_pl_i]
        if Vi >= Vi_bar:
            # res = "STABLE"
            cpt_STABLE_players += 1
            bool_NASH_modeprofile = True
        else:
            # res = "INSTABLE"
            bool_NASH_modeprofile = False
            
    return bool_NASH_modeprofile, cpt_STABLE_players
# ________      checkout nash for mode profile ---> fin          ______________

# ________      classification of Perf_t ---> debut     ______________________
def select_best_bad_mid_modeprofiles_BF_NH( t, Perf_t, cpt_xxx, 
                                           Perf_ts_BF, Perf_ts_NH, 
                                           dico_mode_prof_by_players,
        best_key_Perf_t_BF, dico_modes_profs_by_players_t_bestBF,
        bad_key_Perf_t_BF, dico_modes_profs_by_players_t_badBF,
        mid_key_Perf_t_BF, dico_modes_profs_by_players_t_midBF, 
        bool_NASH_modeprofile, dico_nash, mode_profile,
        best_key_Perf_t_NH, dico_modes_profs_by_players_t_bestNH,
        bad_key_Perf_t_NH, dico_modes_profs_by_players_t_badNH,
        mid_key_Perf_t_NH, dico_modes_profs_by_players_t_midNH, 
        keys_best_BF_NH, keys_bad_BF_NH):
    """
    selection of the best_BF, middle_BF, bad_BF Perf_t modes' profiles and 
    the best_NH, middle_NH, bad_NH Perf_t modes' profiles 
    
    keys_best_BF_NH  : keys of best bf having a nash equilibrium
    keys_bad_BF_NH  : keys of bad bf having a nash equilibrium
    """
    best_key_Perf_t_BF = bf_game.select_perf_t_from_key(
                                best=best_key_Perf_t_BF,
                                current=Perf_t,
                                key="best")
    bad_key_Perf_t_BF = bf_game.select_perf_t_from_key(
                                best=bad_key_Perf_t_BF,
                                current=Perf_t,
                                key="bad")
    # best_key_Perf_t_NH = bf_game.select_perf_t_from_key(
    #                             best=best_key_Perf_t_NH,
    #                             current=Perf_t,
    #                             key="best")
    # bad_key_Perf_t_NH = bf_game.select_perf_t_from_key(
    #                             best=bad_key_Perf_t_NH,
    #                             current=Perf_t,
    #                             key="bad")
    Perf_ts_BF.append(Perf_t)
    id_mid_key_Perf_t = np.argsort(Perf_ts_BF)[len(Perf_ts_BF)//2]
    mid_key_Perf_t_new = Perf_ts_BF[id_mid_key_Perf_t]
    if best_key_Perf_t_BF == Perf_t \
        and best_key_Perf_t_BF not in dico_modes_profs_by_players_t_bestBF:
        dico_modes_profs_by_players_t_bestBF = dict()
        dico_modes_profs_by_players_t_bestBF[Perf_t] \
            = [ ("BF_{}_t_{}".format(cpt_xxx,t), 
                  dico_mode_prof_by_players) ]
    elif best_key_Perf_t_BF == Perf_t \
        and best_key_Perf_t_BF in dico_modes_profs_by_players_t_bestBF:
        dico_modes_profs_by_players_t_bestBF[Perf_t]\
        .append( ("BF_{}_t_{}".format(cpt_xxx,t), 
                  dico_mode_prof_by_players) )
        
    if bad_key_Perf_t_BF == Perf_t \
        and bad_key_Perf_t_BF not in dico_modes_profs_by_players_t_badBF:
        dico_modes_profs_by_players_t_badBF = dict()
        dico_modes_profs_by_players_t_badBF[Perf_t] \
            = [ ("BF_{}_t_{}".format(cpt_xxx,t), 
                  dico_mode_prof_by_players) ]
    elif bad_key_Perf_t_BF == Perf_t \
        and bad_key_Perf_t_BF in dico_modes_profs_by_players_t_badBF:
        dico_modes_profs_by_players_t_badBF[Perf_t]\
        .append( ("BF_{}_t_{}".format(cpt_xxx,t), 
                  dico_mode_prof_by_players) )
    
    mid_key_Perf_t_OLD = mid_key_Perf_t_BF
    if mid_key_Perf_t_BF != mid_key_Perf_t_new:
        dico_modes_profs_by_players_t_midBF = dict()
        dico_modes_profs_by_players_t_midBF[mid_key_Perf_t_new] \
            = [ ("BF_{}_t_{}".format(cpt_xxx,t), 
                  dico_mode_prof_by_players) ]
        mid_key_Perf_t_BF = mid_key_Perf_t_new
    
    
    # NASH EQUILIBRIUM
    if bool_NASH_modeprofile:
        if len(dico_nash) == 0 or Perf_t not in dico_nash:
            dico_nash[Perf_t] = [mode_profile]
        else:
            dico_nash[Perf_t].append(mode_profile)
        
        best_key_Perf_t_NH = bf_game.select_perf_t_from_key(
                                best=best_key_Perf_t_NH,
                                current=Perf_t,
                                key="best")
        bad_key_Perf_t_NH = bf_game.select_perf_t_from_key(
                                    best=bad_key_Perf_t_NH,
                                    current=Perf_t,
                                    key="bad")    
        
        
        Perf_ts_NH.append(Perf_t)
        id_mid_key_Perf_t_NH = np.argsort(Perf_ts_NH)[len(Perf_ts_NH)//2]
        mid_key_Perf_t_new_NH = Perf_ts_NH[id_mid_key_Perf_t_NH]
        
        if best_key_Perf_t_NH == Perf_t \
            and best_key_Perf_t_NH not in dico_modes_profs_by_players_t_bestNH:
            dico_modes_profs_by_players_t_bestNH = dict()
            dico_modes_profs_by_players_t_bestNH[Perf_t] \
                = [ ("NH_{}_t_{}".format(cpt_xxx,t), 
                      dico_mode_prof_by_players) ]
            keys_best_BF_NH.append("BF_{}_t_{}".format(cpt_xxx,t))
        elif best_key_Perf_t_NH == Perf_t \
            and best_key_Perf_t_NH in dico_modes_profs_by_players_t_bestNH:
            dico_modes_profs_by_players_t_bestNH[Perf_t]\
            .append( ("NH_{}_t_{}".format(cpt_xxx,t), 
                      dico_mode_prof_by_players) )
            keys_best_BF_NH.append("BF_{}_t_{}".format(cpt_xxx,t))
            
        if bad_key_Perf_t_NH == Perf_t \
            and bad_key_Perf_t_NH not in dico_modes_profs_by_players_t_badNH:
            dico_modes_profs_by_players_t_badNH = dict()
            dico_modes_profs_by_players_t_badNH[Perf_t] \
                = [ ("NH_{}_t_{}".format(cpt_xxx,t), 
                      dico_mode_prof_by_players) ]
            keys_bad_BF_NH.append("BF_{}_t_{}".format(cpt_xxx,t))
        elif bad_key_Perf_t_NH == Perf_t \
            and bad_key_Perf_t_NH in dico_modes_profs_by_players_t_badNH:
            dico_modes_profs_by_players_t_badNH[Perf_t]\
            .append( ("NH_{}_t_{}".format(cpt_xxx,t), 
                      dico_mode_prof_by_players) )
            keys_bad_BF_NH.append("BF_{}_t_{}".format(cpt_xxx,t))
        
        if mid_key_Perf_t_NH != mid_key_Perf_t_new_NH:
            dico_modes_profs_by_players_t_midNH = dict()
            dico_modes_profs_by_players_t_midNH[mid_key_Perf_t_new_NH] \
                = [ ("NH_{}_t_{}".format(cpt_xxx,t), 
                      dico_mode_prof_by_players) ]
            mid_key_Perf_t_NH = mid_key_Perf_t_new_NH
        
    return Perf_ts_BF, Perf_ts_NH, \
            best_key_Perf_t_BF, dico_modes_profs_by_players_t_bestBF, \
            bad_key_Perf_t_BF, dico_modes_profs_by_players_t_badBF, \
            mid_key_Perf_t_BF, dico_modes_profs_by_players_t_midBF, \
            best_key_Perf_t_NH, dico_modes_profs_by_players_t_bestNH, \
            bad_key_Perf_t_NH, dico_modes_profs_by_players_t_badNH, \
            mid_key_Perf_t_NH, dico_modes_profs_by_players_t_midNH, \
            keys_best_BF_NH, \
            keys_bad_BF_NH
# ________      classification of Perf_t --->       fin     ___________________
  

# ________       balanced players 4 all modes_profiles   ---> debut      ______
def generer_balanced_players_4_modes_profils(arr_pl_M_T_vars_modif, 
                                            m_players, t,
                                            pi_hp_plus, pi_hp_minus,
                                            a, b,
                                            pi_0_plus_t, pi_0_minus_t,
                                            manual_debug, dbg):
    """
    generate the combinaison of all modes' profils and 
    for each modes' profil, balance the players' game and check out the nash equilibrium
    """
    
    best_key_Perf_t_BF, mid_key_Perf_t_BF, bad_key_Perf_t_BF = None, None, None
    best_key_Perf_t_NH, mid_key_Perf_t_NH, bad_key_Perf_t_NH = None, None, None
    
    dico_modes_profs_by_players_t_bestBF = dict() 
    dico_modes_profs_by_players_t_badBF = dict()
    dico_modes_profs_by_players_t_midBF = dict()
    dico_modes_profs_by_players_t_bestNH = dict() 
    dico_modes_profs_by_players_t_badNH = dict()
    dico_modes_profs_by_players_t_midNH = dict()
    possibles_modes = possibles_modes_players_automate(
                            arr_pl_M_T_vars=arr_pl_M_T_vars_modif, 
                            m_players=m_players, t=t)
    print("possibles_modes={}".format(len(possibles_modes)))
    mode_profiles = it.product(*possibles_modes)
    
    # TODO: to DELETE AFTER
    dico_modprofil_b0cO_Perf_t = dict()
    
    dico_nash = dict() # key=Perf_t, mode_profile
    
    Perf_ts_BF, Perf_ts_NH = list(), list()
    keys_best_BF_NH, keys_bad_BF_NH = list(), list()                           # keys_best_BF_NH: keys of best bf having a nash equilibrium, keys_bad_BF_NH : keys of bad bf having a nash equilibrium
    cpt_xxx = 0
    for mode_profile in mode_profiles:
        arr_pl_Mtvars_mode_prof = arr_pl_M_T_vars_modif.copy()
        arr_pl_Mtvars_mode_prof[:,t, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] \
            = mode_profile
        
        # arr_pl_Mtvars_mode_prof \
        #     = balanced_player_game_4_mode_profil(
        #         arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_mode_prof.copy(), 
        #         m_players=m_players, t=t,
        #         dbg=dbg)
            
        # compute pi_sg_{plus,minus}_t, pi_0_{plus,minus}_t, b0_t, c0_t, ben_t, cst_t
        In_sg, Out_sg, b0_t, c0_t = None, None, None, None 
        bens_t, csts_t = None, None 
        pi_sg_plus_t, pi_sg_minus_t = None, None
        dico_gamme_t = dict()
        
        arr_pl_Mtvars_mode_prof, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t \
            = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                    arr_pl_M_T_vars_modif=arr_pl_Mtvars_mode_prof.copy(),
                    mode_profile=mode_profile, t=t,
                    pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                    a=a, b=b,
                    pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                    random_mode=False,
                    manual_debug=manual_debug, dbg=dbg
                    )
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_Mtvars_mode_prof, t)
            
        # compute Perf_t
        bens_csts_t = bens_t - csts_t
        Perf_t = np.sum(bens_csts_t, axis=0)
        
        # checkout nash mode profile
        bool_NASH_modeprofile, cpt_STABLE_players \
            = checkout_nash_notLearnAlgo(
                arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_mode_prof.copy(), 
                mode_profile=mode_profile, bens_csts_t=bens_csts_t,
                m_players=m_players, t=t, 
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t, 
                manual_debug=manual_debug, dbg=dbg)
        
        # TO DO: TO DELETE AFTER
        dico_modprofil_b0cO_Perf_t[mode_profile]={"bO_t":round(b0_t,2), 
                                                  "cO_t":round(c0_t,2),
                                                  "Perf_t":Perf_t}
            
            
        dico_mode_prof_by_players = dict()
        for num_pl_i in range(0, m_players):
            dico_vars = dict()
            Vi = bens_csts_t[num_pl_i]
            dico_vars["Vi"] = round(Vi, 2)
            dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
            dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
            variables = ["set", "state_i", "mode_i", "Pi", "Ci", "Si_max", 
                         "Si_old", "Si", "prod_i", "cons_i", "r_i", 
                         "Si_minus", "Si_plus", "gamma_i"]
            for variable in variables:
                dico_vars[variable] = arr_pl_Mtvars_mode_prof[
                                        num_pl_i, t, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                
            dico_mode_prof_by_players[fct_aux.RACINE_PLAYER
                                        +str(num_pl_i)
                                        +"_t_"+str(t)
                                        +"_"+str(cpt_xxx)] \
                = dico_vars
            
            dico_modprofil_b0cO_Perf_t[mode_profile]\
                [fct_aux.RACINE_PLAYER+str(num_pl_i)] = dico_vars
        
        dico_mode_prof_by_players["bens_t"] = bens_t
        dico_mode_prof_by_players["csts_t"] = csts_t
        dico_mode_prof_by_players["Perf_t"] = round(Perf_t,2)                  # utility of the game
        dico_mode_prof_by_players["b0_t"] = round(b0_t,2)
        dico_mode_prof_by_players["c0_t"] = round(c0_t,2)
        dico_mode_prof_by_players["Out_sg"] = round(Out_sg,2)
        dico_mode_prof_by_players["In_sg"] = round(In_sg,2)
        dico_mode_prof_by_players["pi_sg_plus_t"] = round(pi_sg_plus_t,2)
        dico_mode_prof_by_players["pi_sg_minus_t"] = round(pi_sg_minus_t,2)
        dico_mode_prof_by_players["pi_0_plus_t"] = round(pi_0_plus_t,2)
        dico_mode_prof_by_players["pi_0_minus_t"] = round(pi_0_minus_t,2)
        dico_mode_prof_by_players["mode_profile"] = mode_profile
        
        
        Perf_ts_BF, Perf_ts_NH, \
        best_key_Perf_t_BF, dico_modes_profs_by_players_t_bestBF, \
        bad_key_Perf_t_BF, dico_modes_profs_by_players_t_badBF, \
        mid_key_Perf_t_BF, dico_modes_profs_by_players_t_midBF, \
        best_key_Perf_t_NH, dico_modes_profs_by_players_t_bestNH, \
        bad_key_Perf_t_NH, dico_modes_profs_by_players_t_badNH, \
        mid_key_Perf_t_NH, dico_modes_profs_by_players_t_midNH, \
        keys_best_BF_NH, \
        keys_bad_BF_NH \
           = select_best_bad_mid_modeprofiles_BF_NH( t, Perf_t, cpt_xxx, 
                                                    Perf_ts_BF, Perf_ts_NH,
                dico_mode_prof_by_players,
                best_key_Perf_t_BF, dico_modes_profs_by_players_t_bestBF,
                bad_key_Perf_t_BF, dico_modes_profs_by_players_t_badBF,
                mid_key_Perf_t_BF, dico_modes_profs_by_players_t_midBF, 
                bool_NASH_modeprofile, dico_nash, mode_profile,
                best_key_Perf_t_NH, dico_modes_profs_by_players_t_bestNH,
                bad_key_Perf_t_NH, dico_modes_profs_by_players_t_badNH,
                mid_key_Perf_t_NH, dico_modes_profs_by_players_t_midNH, 
                keys_best_BF_NH, keys_bad_BF_NH
                )
        cpt_xxx += 1
        
        print("cpt_xxx={}, After running free memory={}%".format(cpt_xxx,
                        list(psutil.virtual_memory())[2]    )) \
                if cpt_xxx % MOD ==0 else None
                    
    print("Perf_t BF: BAD={}, MIDDLE={}, BEST={}".format(
        bad_key_Perf_t_BF, mid_key_Perf_t_BF, best_key_Perf_t_BF))
    print("Perf_t NH: BAD={}, MIDDLE={}, BEST={}".format(
        bad_key_Perf_t_NH, mid_key_Perf_t_NH, best_key_Perf_t_NH))
 
    
    list_dico_modes_profs_by_players_t_bestBF \
        = dico_modes_profs_by_players_t_bestBF[best_key_Perf_t_BF]
    list_dico_modes_profs_by_players_t_badBF \
        = dico_modes_profs_by_players_t_badBF[bad_key_Perf_t_BF]
    list_dico_modes_profs_by_players_t_midBF \
        = dico_modes_profs_by_players_t_midBF[mid_key_Perf_t_BF]
        
    list_dico_modes_profs_by_players_t_bestNH \
        = dico_modes_profs_by_players_t_bestNH[best_key_Perf_t_NH]
    list_dico_modes_profs_by_players_t_badNH \
        = dico_modes_profs_by_players_t_badNH[bad_key_Perf_t_NH]
    list_dico_modes_profs_by_players_t_midNH \
        = dico_modes_profs_by_players_t_midNH[mid_key_Perf_t_NH]
        
    set_Perf_ts_BF = set(Perf_ts_BF)
        
    return list_dico_modes_profs_by_players_t_bestBF, \
            list_dico_modes_profs_by_players_t_badBF, \
            list_dico_modes_profs_by_players_t_midBF, \
            list_dico_modes_profs_by_players_t_bestNH, \
            list_dico_modes_profs_by_players_t_badNH, \
            list_dico_modes_profs_by_players_t_midNH, \
            keys_best_BF_NH, keys_bad_BF_NH, \
            set_Perf_ts_BF, dico_modprofil_b0cO_Perf_t                          # TODO TO DELETE dico_modprofil_b0cO_Perf_t
        
# ________       balanced players 4 all modes_profiles   ---> fin        ______


# ____  test values btw arr_pl_Mtvars_algo and dicoModeProfByPlayers : debut __ 
def compute_prices_inside_SG(arr_pl_Mtvars_algo, m_players, t,
                             pi_hp_plus, pi_hp_minus, a, b,
                             pi_0_plus_t, pi_0_minus_t,
                             manual_debug, dbg):
    arr_pl_Mtvars_algo_modif = balanced_player_game_4_mode_profil(
                                arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_algo, 
                                m_players=m_players, t=t,
                                dbg=dbg)
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t \
        = fct_aux.compute_prices_inside_SG_4_notLearnAlgo(
            arr_pl_M_T_vars_modif=arr_pl_Mtvars_algo_modif, t=t,
            pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
            a=a, b=b,
            pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
            manual_debug=manual_debug, dbg=dbg)
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_Mtvars_algo_modif, t)
    
    return In_sg, Out_sg, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t
    
    
def test_same_values_dicoModeProfByPlayers(arr_pl_Mtvars_algo, 
                                           m_players, t, 
                                           pi_hp_plus, pi_hp_minus, 
                                           a, b, 
                                           pi_0_plus_t, pi_0_minus_t, 
                                           bens_t, csts_t, b0_t, c0_t,
                                           manual_debug, dbg):
    """
    test if there are the same values like these in dico_mode_prof_by_players
    """
    In_sg_new, Out_sg_new, b0_t_new, c0_t_new = None, None, None, None
    bens_t_new, csts_t_new = None, None
    pi_sg_plus_t_new, pi_sg_minus_t_new = None, None
    In_sg_new, Out_sg_new, \
    b0_t_new, c0_t_new, \
    bens_t_new, csts_t_new, \
    pi_sg_plus_t_new, pi_sg_minus_t_new \
        = compute_prices_inside_SG(
                arr_pl_Mtvars_algo=arr_pl_Mtvars_algo, 
                m_players=m_players, t=t, 
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                manual_debug=manual_debug, dbg=dbg)
    bens_csts_t_new = bens_t_new - csts_t_new
    Perf_t_new = np.sum(bens_csts_t_new, axis=0)
    
    
    # compute Perf_t
    bens_csts_t = bens_t - csts_t
    Perf_t = np.sum(bens_csts_t, axis=0)
    ## _______
    
    ##### verification of best key quality 
    diff = np.abs(Perf_t_new - Perf_t)
    print(" Perf_t_algo == Perf_t_new --> OK (diff={}) ".format(diff)) \
        if diff < 0.1 \
        else print("Perf_t_algo != Perf_t_new --> NOK (diff={}) \n"\
                   .format(diff))     
# ____  test values btw arr_pl_Mtvars_algo and dicoModeProfByPlayers : debut __  


# ____________        checkout NASH equilibrium --> debut        ______________
def checkout_nash_4_profils_by_OnePeriodt(arr_pl_Mtvars_modif_algo,
                                          arr_pl_M_t_vars_init,
                                          pi_hp_plus, pi_hp_minus, a, b,
                                          pi_0_minus_t, pi_0_plus_t, 
                                          bens_csts_M_t,
                                          m_players, t,
                                          manual_debug, dbg):
    """
    verify if the modes' profil of players at time t is a Nash equilibrium.
    """
    # create a result dataframe of checking players' stability and nash equilibrium
    cols = ["players", "nash_modes_t{}".format(t), 'states_t{}'.format(t), 
            'Vis_t{}'.format(t), 'Vis_bar_t{}'.format(t), 
               'res_t{}'.format(t)] 
    
    arr_pl_Mtvars_modif_algo = arr_pl_Mtvars_modif_algo.copy()
    
    id_players = list(range(0, m_players))
    df_nash_t = pd.DataFrame(index=id_players, columns=cols)
    
    # revert Si to the initial value ie at t and k=0
    Sis_init = arr_pl_M_t_vars_init[:, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
    arr_pl_Mtvars_modif_algo[:, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]] = Sis_init
    
    # stability of each player
    modes_profil = list(arr_pl_Mtvars_modif_algo[
                            :, t,
                            fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] )
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_Mtvars_modif_algo[
                        num_pl_i, t,
                        fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] 
        mode_i = modes_profil[num_pl_i]
        mode_i_bar = fct_aux.find_out_opposite_mode(state_i, mode_i)
        
        opposite_modes_profil = modes_profil.copy()
        opposite_modes_profil[num_pl_i] = mode_i_bar
        opposite_modes_profil = tuple(opposite_modes_profil)
        arr_pl_Mtvars_modif_algo[:, t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] \
            = opposite_modes_profil 
        
        df_nash_t.loc[num_pl_i, "players"] = fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)
        df_nash_t.loc[num_pl_i, "nash_modes_t{}".format(t)] = mode_i
        df_nash_t.loc[num_pl_i, "states_t{}".format(t)] = state_i
        
        arr_pl_Mtvars_modif_algo = balanced_player_game_4_mode_profil(
                                    arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_modif_algo, 
                                    m_players=m_players, 
                                    t=t, dbg=dbg)
        ## test if there are the same values like these in dico_mode_prof_by_players
        In_sg_bar, Out_sg_bar, b0_t_bar, c0_t_bar = None, None, None, None
        bens_t_bar, csts_t_bar = None, None
        pi_sg_plus_t_bar, pi_sg_minus_t_bar = None, None
        In_sg_bar, Out_sg_bar, \
        b0_t_bar, c0_t_bar, \
        bens_t_bar, csts_t_bar, \
        pi_sg_plus_t_bar, pi_sg_minus_t_bar \
            = compute_prices_inside_SG(
                arr_pl_Mtvars_algo=arr_pl_Mtvars_modif_algo, 
                m_players=m_players, t=t,
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                manual_debug=manual_debug, dbg=dbg)
            
        bens_csts_t_bar = bens_t_bar - csts_t_bar
        
        Vi = bens_csts_M_t[num_pl_i]
        Vi_bar = bens_csts_t_bar[num_pl_i]
        
        df_nash_t.loc[num_pl_i, 'Vis_t{}'.format(t)] = Vi
        df_nash_t.loc[num_pl_i, 'Vis_bar_t{}'.format(t)] = Vi_bar
        res = None
        if Vi >= Vi_bar:
            res = "STABLE"
            df_nash_t.loc[num_pl_i, 'res_t{}'.format(t)] = res
        else:
            res = "INSTABLE"
            df_nash_t.loc[num_pl_i, 'res_t{}'.format(t)] = res   
            
    return df_nash_t
    
# ____________        checkout NASH equilibrium --> fin          ______________


###############################################################################
#                   definition  des fonctions principales
#
###############################################################################

# __________       main function of brute force, Nash   ---> debut       ______
def BF_NASH_balanced_player_game(arr_pl_M_T_vars_init, algo_names,
                                pi_hp_plus=0.02, 
                                pi_hp_minus=0.33,
                                a=1, b=1,
                                gamma_version=1,
                                path_to_save="tests", 
                                name_dir="tests", 
                                date_hhmm="DDMM_HHMM",
                                manual_debug=False, 
                                criteria_bf="Perf_t", dbg=False):
    """
     run brute force and Nash detection algorithms.

    Returns
    -------
    None.

    """
    
    print("\n \n BF , NASH games: pi_hp_plus={}, pi_hp_minus={} ---> debut \n"\
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
    
    dico_profils_BF, dico_best_profils_BF, dico_bad_profils_BF = dict(), dict(), dict()
    dico_profils_NH, dico_best_profils_NH, dico_bad_profils_NH = dict(), dict(), dict()
    
    dico_modes_profs_by_players_t = dict()
    t = 0
    
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
    print("Sis = {} \n".format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
    
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
            = generer_balanced_players_4_modes_profils(
                arr_pl_M_T_vars_modif, 
                m_players, t,
                pi_hp_plus, pi_hp_minus,
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
    # dico_profils_NH["nb_profils"] = len(list_dico_modes_profs_by_players_t_bestNH) \
    #                                 + len(list_dico_modes_profs_by_players_t_badNH) \
    #                                 + len(list_dico_modes_profs_by_players_t_midNH)    
            
    for algo_name in algo_names:
        
        print("*** algo_name={} *** ".format(algo_name))
        
        list_dico_modes_profs_by_players_t = dict()
        
        if algo_name == fct_aux.ALGO_NAMES_BF[0]:                              # BEST-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_bestBF.copy()
            dico_best_profils_BF["nb_best_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_best_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_best_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
                key_best_BF_cptxxx = tu_key_dico[0]
                if key_best_BF_cptxxx in keys_best_BF_NH:
                    dico_best_profils_BF["nashs"].append(tu_key_dico[1]["mode_profile"])
                    dico_best_profils_BF["Perfs_nash"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[1]:                            # BAD-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_badBF.copy()
            dico_bad_profils_BF["nb_bad_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_bad_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_bad_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
                key_bad_BF_cptxxx = tu_key_dico[0]
                if key_bad_BF_cptxxx in keys_bad_BF_NH:
                    dico_bad_profils_BF["nashs"].append(tu_key_dico[1]["mode_profile"])
                    dico_bad_profils_BF["Perfs_nash"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[2]:                            # MIDDLE-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_midBF.copy()
            dico_mid_profils_BF["nb_mid_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_mid_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_mid_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[0]:                          # BEST-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_bestNH.copy()
            dico_best_profils_NH["nb_best_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_best_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_best_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                          # BAD-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_badNH.copy()
            dico_bad_profils_NH["nb_bad_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_bad_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_bad_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
                
        elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                          # MIDDLE-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_midNH.copy()
            dico_mid_profils_NH["nb_mid_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_mid_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_mid_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
                
        rd_key = None
        if len(list_dico_modes_profs_by_players_t) == 0:
            continue
        elif len(list_dico_modes_profs_by_players_t) == 1:
            rd_key = 0
        else:
            rd_key \
                = np.random.randint(0, len(list_dico_modes_profs_by_players_t))
        id_cpt_xxx, dico_mode_prof_by_players_algo \
            = list_dico_modes_profs_by_players_t[rd_key] 
        print("rd_key={}, cpt_xxx={}".format(rd_key, id_cpt_xxx))
        
        bens_t = dico_mode_prof_by_players_algo["bens_t"]
        csts_t = dico_mode_prof_by_players_algo["csts_t"]
        Perf_t = dico_mode_prof_by_players_algo["Perf_t"]
        b0_t = dico_mode_prof_by_players_algo["b0_t"]
        c0_t = dico_mode_prof_by_players_algo["c0_t"]
        Out_sg = dico_mode_prof_by_players_algo["Out_sg"]
        In_sg = dico_mode_prof_by_players_algo["In_sg"]
        pi_sg_plus_t = dico_mode_prof_by_players_algo["pi_sg_plus_t"]
        pi_sg_minus_t = dico_mode_prof_by_players_algo["pi_sg_minus_t"]
        pi_0_plus_t = dico_mode_prof_by_players_algo["pi_0_plus_t"]
        pi_0_minus_t = dico_mode_prof_by_players_algo["pi_0_minus_t"]
        mode_profile = dico_mode_prof_by_players_algo["mode_profile"]
        
        arr_pl_Mtvars_algo = arr_pl_M_T_vars_modif.copy()
        arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] \
            = mode_profile
        print("mode_profile={}, mode_is={}".format(mode_profile, 
                arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]))
        print("state_is={} ".format( 
                arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]))
        
        # arr_pl_Mtvars_algo = bf_game.balanced_player_game_4_mode_profil(
        #                                  arr_pl_Mtvars_algo, 
        #                                  m_players,
        #                                  dbg)
        arr_pl_Mtvars_algo = balanced_player_game_4_mode_profil(
                                arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_algo, 
                                m_players=m_players, t=t,
                                dbg=dbg)
        
        ## test if there are the same values like these in dico_mode_prof_by_players_algo
        test_same_values_dicoModeProfByPlayers(arr_pl_Mtvars_algo, 
                                               m_players, t, 
                                               pi_hp_plus, pi_hp_minus, 
                                               a, b, 
                                               pi_0_plus_t, pi_0_minus_t, 
                                               bens_t, csts_t, b0_t, c0_t,
                                               manual_debug, dbg)
        print("b0_t={}, c0_t={}, Out_sg={},In_sg={}, pi_hp_minus_t={}, pi_hp_plus_t={}\n".format(
                    b0_t, c0_t, Out_sg, In_sg, pi_hp_minus_t, pi_hp_plus_t))
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
            
        # checkout NASH equilibrium
        df_nash = None
        bens_csts_M_t = bens_t - csts_t
        df_nash = checkout_nash_4_profils_by_OnePeriodt(
                        arr_pl_Mtvars_algo.copy(),
                        arr_pl_M_T_vars_init,
                        pi_hp_plus, pi_hp_minus, a, b,
                        pi_0_minus_t, pi_0_plus_t, 
                        bens_csts_M_t,
                        m_players,
                        t,
                        manual_debug,
                        dbg)
        # if algo_name not in fct_aux.ALGO_NAMES_NASH:
        #     bens_csts_M_t = bens_t - csts_t
        #     df_nash = checkout_nash_4_profils_by_OnePeriodt(
        #                     arr_pl_Mtvars_algo.copy(),
        #                     arr_pl_M_T_vars_init,
        #                     pi_hp_plus, pi_hp_minus, a, b,
        #                     pi_0_minus_t, pi_0_plus_t, 
        #                     bens_csts_M_t,
        #                     m_players,
        #                     t,
        #                     manual_debug,
        #                     dbg)
        
        # ___________ update saving variables : debut ______________________
        #arr_pl_M_T_vars_modif[:,t,:] = arr_pl_Mtvars_algo
        b0_ts_T_algo, c0_ts_T_algo, \
        BENs_M_T_algo, CSTs_M_T_algo, \
        pi_sg_plus_T_algo, pi_sg_minus_T_algo, \
        pi_0_plus_T_algo, pi_0_minus_T_algo, \
        df_nash_algo \
            = bf_game.update_saving_variables(t, 
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
        dico_modes_profs_by_players_t[t] = dico_mode_prof_by_players_algo
        #___________ update saving variables : fin   ______________________
        
        print("Sis = {}".format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        print("----- t={} After running free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
        
        # __________        compute prices variables         ____________________
        B_is_M_algo, C_is_M_algo, BB_is_M_algo, CC_is_M_algo, EB_is_M_algo, \
        B_is_MT_cum_algo, C_is_MT_cum_algo, \
        B_is_M_T_algo, C_is_M_T_algo, BB_is_M_T_algo, CC_is_M_T_algo, \
        EB_is_M_T_algo \
            = fct_aux.compute_prices_B_C_BB_CC_EB_DET(
                    arr_pl_M_T_vars_modif=arr_pl_Mtvars_algo, 
                    pi_sg_minus_T=pi_sg_minus_T_algo, pi_sg_plus_T=pi_sg_plus_T_algo, 
                    pi_0_minus_T=pi_0_minus_T_algo, pi_0_plus_T=pi_0_plus_T_algo,
                    b0_s_T=b0_ts_T_algo, c0_s_T=c0_ts_T_algo)
            
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
        # else:
        #     path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
        #                                 msg, algo_name
        #                                 )
        
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
                arr_pl_M_T_K_vars=arr_pl_Mtvars_algo, 
                b0_s_T_K=b0_ts_T_algo, c0_s_T_K=c0_ts_T_algo, 
                B_is_M=B_is_M_algo, C_is_M=C_is_M_algo, 
                B_is_M_T=B_is_M_T_algo, C_is_M_T=C_is_M_T_algo,
                BENs_M_T_K=BENs_M_T_algo, CSTs_M_T_K=CSTs_M_T_algo, 
                BB_is_M=BB_is_M_algo, CC_is_M=CC_is_M_algo, EB_is_M=EB_is_M_algo, 
                BB_is_M_T=BB_is_M_T_algo, CC_is_M_T=CC_is_M_T_algo, 
                EB_is_M_T=EB_is_M_T_algo,
                dico_EB_R_EBsetA1B1_EBsetB2C=dico_EB_R_EBsetA1B1_EBsetB2C,
                pi_sg_minus_T_K=pi_sg_minus_T_algo, pi_sg_plus_T_K=pi_sg_plus_T_algo, 
                pi_0_minus_T_K=pi_0_minus_T_algo, pi_0_plus_T_K=pi_0_plus_T_algo,
                pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T, 
                dico_stats_res=dico_modes_profs_by_players_t, 
                algo=algo_name, 
                dico_best_steps=dict())
        # bf_game.turn_dico_stats_res_into_df_BF(
        #       dico_modes_profs_players_algo = dico_modes_profs_by_players_t, 
        #       path_to_save = path_to_save, 
        #       t_periods = t_periods, 
        #       manual_debug = manual_debug, 
        #       algo_name = algo_name)
    
    return dico_profils_BF, dico_profils_NH, \
            dico_best_profils_BF, dico_bad_profils_BF, dico_mid_profils_BF, \
            dico_best_profils_NH, dico_bad_profils_NH, dico_mid_profils_NH

   
# __________       main function of brute force, Nash   ---> fin         ______

def bf_nash_game_model_1t_LOOK4BadBestMid(
                        list_dico_modes_profs_by_players_t_bestBF,
                        list_dico_modes_profs_by_players_t_badBF,
                        list_dico_modes_profs_by_players_t_midBF,
                        list_dico_modes_profs_by_players_t_bestNH,
                        list_dico_modes_profs_by_players_t_badNH,
                        list_dico_modes_profs_by_players_t_midNH,
                        keys_best_BF_NH, keys_bad_BF_NH, set_Perf_ts_BF, dico_modprofil_b0cO_Perf_t,
                        pi_hp_plus_T, pi_hp_minus_T,
                        m_players, t_periods,
                        arr_pl_M_T_vars_init,
                        arr_pl_MTvars_modif, t,
                        algos_BF_NH,
                        pi_hp_plus, 
                        pi_hp_minus,
                        a, b,
                        gamma_version,
                        path_to_save, 
                        name_dir, 
                        date_hhmm,
                        manual_debug, 
                        criteria_bf, dbg):
    
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
    
    m_players = arr_pl_MTvars_modif.shape[0]
    
    for algo_name in algos_BF_NH:
        
        pi_sg_plus_T = np.empty(shape=(t_periods,)); pi_sg_plus_T.fill(np.nan)  # shape (T_PERIODS,)
        pi_sg_minus_T = np.empty(shape=(t_periods,)); pi_sg_plus_T.fill(np.nan) # shape (T_PERIODS,)
        pi_0_plus_T = np.empty(shape=(t_periods,)); pi_0_plus_T.fill(np.nan)    # shape (T_PERIODS,)
        pi_0_minus_T = np.empty(shape=(t_periods,)); pi_0_minus_T.fill(np.nan)  # shape (T_PERIODS,)
        B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)              # shape (M_PLAYERS, )
        C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)              # shape (M_PLAYERS, )
        B_is_M_T = np.empty(shape=(m_players, t_periods)); B_is_M_T.fill(np.nan)# shape (M_PLAYERS, )
        C_is_M_T = np.empty(shape=(m_players, t_periods)); C_is_M_T.fill(np.nan)# shape (M_PLAYERS, )
        b0_ts_T = np.empty(shape=(t_periods,)); b0_ts_T.fill(np.nan)            # shape (T_PERIODS,)
        c0_ts_T = np.empty(shape=(t_periods,)); c0_ts_T.fill(np.nan)
        BENs_M_T = np.empty(shape=(m_players, t_periods))                       # shape (M_PLAYERS, T_PERIODS)
        CSTs_M_T = np.empty(shape=(m_players, t_periods))
        CC_is_M = np.empty(shape=(m_players,)); CC_is_M.fill(np.nan)            # shape (M_PLAYERS, )
        BB_is_M = np.empty(shape=(m_players,)); BB_is_M.fill(np.nan)            # shape (M_PLAYERS, )
        EB_is_M = np.empty(shape=(m_players,)); EB_is_M.fill(np.nan)            # shape (M_PLAYERS, )
        CC_is_M_T = np.empty(shape=(m_players, t_periods)); CC_is_M_T.fill(np.nan)# shape (M_PLAYERS, )
        BB_is_M_T = np.empty(shape=(m_players, t_periods)); BB_is_M_T.fill(np.nan)# shape (M_PLAYERS, )
        EB_is_M_T = np.empty(shape=(m_players, t_periods)); EB_is_M_T.fill(np.nan)# shape (M_PLAYERS, )
    
        dico_modes_profs_by_players_t = dict()    
    
        print("\n *** algo_name={} *** ".format(algo_name))
        
        
        list_dico_modes_profs_by_players_t = dict()
        
        if algo_name == fct_aux.ALGO_NAMES_BF[0]:                              # BEST-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_bestBF.copy()
            dico_best_profils_BF["nb_best_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_best_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_best_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
                key_best_BF_cptxxx = tu_key_dico[0]
                if key_best_BF_cptxxx in keys_best_BF_NH:
                    dico_best_profils_BF["nashs"].append(tu_key_dico[1]["mode_profile"])
                    dico_best_profils_BF["Perfs_nash"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[1]:                            # BAD-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_badBF.copy()
            dico_bad_profils_BF["nb_bad_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_bad_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_bad_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
                key_bad_BF_cptxxx = tu_key_dico[0]
                if key_bad_BF_cptxxx in keys_bad_BF_NH:
                    dico_bad_profils_BF["nashs"].append(tu_key_dico[1]["mode_profile"])
                    dico_bad_profils_BF["Perfs_nash"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[2]:                            # MIDDLE-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_midBF.copy()
            dico_mid_profils_BF["nb_mid_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_mid_profils_BF["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_mid_profils_BF["Perfs"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[0]:                          # BEST-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_bestNH.copy()
            dico_best_profils_NH["nb_best_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_best_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_best_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:                          # BAD-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_badNH.copy()
            dico_bad_profils_NH["nb_bad_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_bad_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_bad_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
                
        elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:                          # MIDDLE-NASH
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_midNH.copy()
            dico_mid_profils_NH["nb_mid_profils"] = len(list_dico_modes_profs_by_players_t)
            for tu_key_dico in list_dico_modes_profs_by_players_t:
                dico_mid_profils_NH["profils"].append(tu_key_dico[1]["mode_profile"])
                dico_mid_profils_NH["Perfs"].append(tu_key_dico[1]["Perf_t"])
                
        rd_key = None
        if len(list_dico_modes_profs_by_players_t) == 0:
            continue
        elif len(list_dico_modes_profs_by_players_t) == 1:
            rd_key = 0
        else:
            rd_key \
                = np.random.randint(0, len(list_dico_modes_profs_by_players_t))
        id_cpt_xxx, dico_mode_prof_by_players_algo \
            = list_dico_modes_profs_by_players_t[rd_key] 
        print("rd_key={}, cpt_xxx={}".format(rd_key, id_cpt_xxx))
        
        bens_t = dico_mode_prof_by_players_algo["bens_t"]
        csts_t = dico_mode_prof_by_players_algo["csts_t"]
        Perf_t = dico_mode_prof_by_players_algo["Perf_t"]
        b0_t = dico_mode_prof_by_players_algo["b0_t"]
        c0_t = dico_mode_prof_by_players_algo["c0_t"]
        Out_sg = dico_mode_prof_by_players_algo["Out_sg"]
        In_sg = dico_mode_prof_by_players_algo["In_sg"]
        pi_sg_plus_t = dico_mode_prof_by_players_algo["pi_sg_plus_t"]
        pi_sg_minus_t = dico_mode_prof_by_players_algo["pi_sg_minus_t"]
        pi_0_plus_t = dico_mode_prof_by_players_algo["pi_0_plus_t"]
        pi_0_minus_t = dico_mode_prof_by_players_algo["pi_0_minus_t"]
        mode_profile = dico_mode_prof_by_players_algo["mode_profile"]
        
        arr_pl_Mtvars_algo = arr_pl_MTvars_modif.copy()
        arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] \
            = mode_profile
        print("mode_profile={}, mode_is={}".format(mode_profile, 
                arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]))
        print("state_is={} ".format( 
                arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]))
        
        arr_pl_Mtvars_algo = balanced_player_game_4_mode_profil(
                                arr_pl_Mtvars_mode_prof=arr_pl_Mtvars_algo.copy(), 
                                m_players=m_players, t=t,
                                dbg=dbg)
        
        ## test if there are the same values like these in dico_mode_prof_by_players_algo
        test_same_values_dicoModeProfByPlayers(arr_pl_Mtvars_algo.copy(), 
                                               m_players, t, 
                                               pi_hp_plus, pi_hp_minus, 
                                               a, b, 
                                               pi_0_plus_t, pi_0_minus_t, 
                                               bens_t, csts_t, b0_t, c0_t,
                                               manual_debug, dbg)
        print("b0_t={}, c0_t={}, Out_sg={},In_sg={}, pi_hp_minus_t={}, pi_hp_plus_t={}\n".format(
                    b0_t, c0_t, Out_sg, In_sg, pi_hp_minus_T[t], pi_hp_plus_T[t]))
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
            
        # checkout NASH equilibrium
        df_nash = None
        bens_csts_M_t = bens_t - csts_t ; 
        df_nash = checkout_nash_4_profils_by_OnePeriodt(
                        arr_pl_Mtvars_algo.copy(),
                        arr_pl_M_T_vars_init.copy(),
                        pi_hp_plus, pi_hp_minus, a, b,
                        pi_0_minus_t, pi_0_plus_t, 
                        bens_csts_M_t,
                        m_players,
                        t,
                        manual_debug,
                        dbg)
        # df_nash = checkout_nash_4_profils_by_OnePeriodt(
        #                 arr_pl_Mtvars_algo.copy(),
        #                 arr_pl_MTvars_modif.copy(),
        #                 pi_hp_plus, pi_hp_minus, a, b,
        #                 pi_0_minus_t, pi_0_plus_t, 
        #                 bens_csts_M_t,
        #                 m_players,
        #                 t,
        #                 manual_debug,
        #                 dbg)
        # if algo_name not in fct_aux.ALGO_NAMES_NASH:
        #     bens_csts_M_t = bens_t - csts_t
        #     df_nash = checkout_nash_4_profils_by_OnePeriodt(
        #                     arr_pl_Mtvars_algo.copy(),
        #                     arr_pl_M_T_vars_init,
        #                     pi_hp_plus, pi_hp_minus, a, b,
        #                     pi_0_minus_t, pi_0_plus_t, 
        #                     bens_csts_M_t,
        #                     m_players,
        #                     t,
        #                     manual_debug,
        #                     dbg)
        
        # ___________ update saving variables : debut ______________________
        #arr_pl_M_T_vars_modif[:,t,:] = arr_pl_Mtvars_algo
        b0_ts_T_algo, c0_ts_T_algo, \
        BENs_M_T_algo, CSTs_M_T_algo, \
        pi_sg_plus_T_algo, pi_sg_minus_T_algo, \
        pi_0_plus_T_algo, pi_0_minus_T_algo, \
        df_nash_algo \
            = bf_game.update_saving_variables(t, 
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
        dico_modes_profs_by_players_t[t] = dico_mode_prof_by_players_algo
        #___________ update saving variables : fin   ______________________
        
        print("Sis_modif = {}".format(arr_pl_MTvars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        print("Apres exec: Sis_modif_tvars_algo = {}".format(arr_pl_Mtvars_algo[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        print("----- t={} After running free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
        
        # __________        compute prices variables         ____________________
        B_is_M_algo, C_is_M_algo, BB_is_M_algo, CC_is_M_algo, EB_is_M_algo, \
        B_is_MT_cum_algo, C_is_MT_cum_algo, \
        B_is_M_T_algo, C_is_M_T_algo, BB_is_M_T_algo, CC_is_M_T_algo, \
        EB_is_M_T_algo \
            = fct_aux.compute_prices_B_C_BB_CC_EB_DET(
                    arr_pl_M_T_vars_modif=arr_pl_Mtvars_algo.copy(), 
                    pi_sg_minus_T=pi_sg_minus_T_algo, pi_sg_plus_T=pi_sg_plus_T_algo, 
                    pi_0_minus_T=pi_0_minus_T_algo, pi_0_plus_T=pi_0_plus_T_algo,
                    b0_s_T=b0_ts_T_algo, c0_s_T=c0_ts_T_algo)
            
        dico_EB_R_EBsetA1B1_EBsetB2C = {"EB_setA1B1":[np.nan],"EB_setB2C":[np.nan], 
                                        "ER":[np.nan], "VR":[np.nan]}
        # __________        compute prices variables         ____________________
        
        #_______      save computed variables locally from algo_name     __________
        msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
        
        if "simu_DDMM_HHMM" in path_to_save:
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                        msg, algo_name
                                        )
        # else:
        #     path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
        #                                 msg, algo_name
        #                                 )
        print("path_to_save={}".format(path_to_save))
        
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        # df_nash.to_excel(os.path.join(
        #             *[path_to_save,
        #               "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
        #             index=False)
        df_nash.to_csv(os.path.join(
                    *[path_to_save,
                      "resume_verify_Nash_equilibrium_{}.csv".format(algo_name)]), 
                    index=False)
        
        np.save(os.path.join(path_to_save, "arr_pl_M_T_K_vars_init.npy"), 
                arr_pl_M_T_vars_init)
        fct_aux.save_variables(
                path_to_save=path_to_save, 
                arr_pl_M_T_K_vars=arr_pl_Mtvars_algo, 
                b0_s_T_K=b0_ts_T_algo, c0_s_T_K=c0_ts_T_algo, 
                B_is_M=B_is_M_algo, C_is_M=C_is_M_algo, 
                B_is_M_T=B_is_M_T_algo, C_is_M_T=C_is_M_T_algo,
                BENs_M_T_K=BENs_M_T_algo, CSTs_M_T_K=CSTs_M_T_algo, 
                BB_is_M=BB_is_M_algo, CC_is_M=CC_is_M_algo, EB_is_M=EB_is_M_algo, 
                BB_is_M_T=BB_is_M_T_algo, CC_is_M_T=CC_is_M_T_algo, 
                EB_is_M_T=EB_is_M_T_algo,
                dico_EB_R_EBsetA1B1_EBsetB2C=dico_EB_R_EBsetA1B1_EBsetB2C,
                pi_sg_minus_T_K=pi_sg_minus_T_algo, pi_sg_plus_T_K=pi_sg_plus_T_algo, 
                pi_0_minus_T_K=pi_0_minus_T_algo, pi_0_plus_T_K=pi_0_plus_T_algo,
                pi_hp_plus_T=pi_hp_plus_T, pi_hp_minus_T=pi_hp_minus_T, 
                dico_stats_res=dico_modes_profs_by_players_t, 
                algo=algo_name, 
                dico_best_steps=dict())
        
        # save solutions found in algo_name
        dico_key_dico = dict()
        for tu_key_dico in list_dico_modes_profs_by_players_t:
            tu_key_dico[1]["set_all_Perf_t"] = set_Perf_ts_BF
            tu_key_dico[1]["modprofil_b0cO_Perf_t"] = dico_modprofil_b0cO_Perf_t
            dico_key_dico[tu_key_dico[0]] = tu_key_dico[1]
        pd.DataFrame(dico_key_dico).to_csv(os.path.join(
                    *[path_to_save,
                      "resume_solutions_{}.csv".format(algo_name)]), 
                    index=True)
        
    dico_profils_NH["nb_profils"] = set(dico_best_profils_NH["profils"])\
                                    .union( set(dico_bad_profils_NH["profils"])\
                                           .union(set(dico_mid_profils_NH["profils"])))
                                        
    return dico_profils_BF, dico_profils_NH, \
            dico_best_profils_BF, dico_bad_profils_BF, dico_mid_profils_BF, \
            dico_best_profils_NH, dico_bad_profils_NH, dico_mid_profils_NH

###############################################################################
#                   definition  des unittests
#
###############################################################################
def test_BF_NASH_balanced_player_game():
    fct_aux.N_DECIMALS = 8
    a, b = 1, 1
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 20 #0.33 #[15, 5]
    
    manual_debug = False
    gamma_version = -2  #-2,-1,1,2,3,4,5
    debug = False
    criteria_bf = "Perf_t"
    used_instances = True #False#True
    
    t_periods = 1
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    
    # _____                     scenarios --> debut                 __________
    scenario_name = "scenarioOnePeriod"      
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
    setA_m_players, setB_m_players, setC_m_players = 4, 3, 3                   # 10 players
    arr_pl_M_T_vars_init \
        = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc24(
            setA_m_players=setA_m_players, 
            setB_m_players=setB_m_players, 
            setC_m_players=setC_m_players, 
            t_periods=t_periods, 
            scenario=None,
            scenario_name=scenario_name,
            path_to_arr_pl_M_T=path_to_arr_pl_M_T, 
            used_instances=used_instances)
        
    # _____                     scenarios --> fin                   __________
    
    algo_names = fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH 
    
    global MOD
    m_players = setA_m_players + setB_m_players + setC_m_players
    MOD = int(0.10*pow(2, m_players)) \
            if pow(2, m_players) < 65000 \
            else int(0.020*pow(2, m_players))    
    
    name_simu = "BF_NASH_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    dico_profils_BF, dico_profils_NH, \
    dico_best_profils_BF, dico_bad_profils_BF, dico_mid_profils_BF, \
    dico_best_profils_NH, dico_bad_profils_NH, dico_mid_profils_NH \
       = BF_NASH_balanced_player_game(
            arr_pl_M_T_vars_init=arr_pl_M_T_vars_init.copy(), 
            algo_names=algo_names,
            pi_hp_plus=pi_hp_plus, 
            pi_hp_minus=pi_hp_minus,
            a=a, b=b,
            gamma_version=gamma_version,
            path_to_save=path_to_save, 
            name_dir="tests", 
            date_hhmm="DDMM_HHMM",
            manual_debug=manual_debug, 
            criteria_bf=criteria_bf, dbg=debug)
    
    
    C1 = True if dico_profils_NH["nb_profils"] > 0 else False
    C6s = dico_best_profils_BF["Perfs"] if dico_best_profils_BF["nb_best_profils"] > 0 else []
    C9s = dico_bad_profils_BF["Perfs"] if dico_bad_profils_BF["nb_bad_profils"] > 0 else []
    C7s_bad, C7s_best = [], []
    if C1:
       C7s_bad = dico_bad_profils_NH["Perfs"] if dico_bad_profils_NH["nb_bad_profils"] > 0 else []
       C7s_best = dico_best_profils_NH["Perfs"] if dico_best_profils_NH["nb_best_profils"] > 0 else []
       
    print("C1 = {}".format(C1))
    print("C6s={}, C9s={}".format(len(C6s), len(C9s) ))
    print("C7s_bad={}, C7s_best={}".format(len(C7s_bad), len(C7s_best) ))
       
    # C5 = Perf_sum_Vi_LRI2
    # C6 = Perf_best_profils_bf[0] if nb_best_profils_bf > 0 else None
    # C9 = Perf_bad_profils_bf[0] if nb_bad_profils_bf > 0 else None
    # if C1:
    #     C7 = Perf_bad_profils_NH[0] if nb_bad_profils_NH > 0 else None
        
    # check_C5_inf_C6 = None
    # if C5 <= C6 and C5 is not None and C6 is not None:
    #     check_C5_inf_C6 = "OK"
    # else:
    #     check_C5_inf_C6 = "NOK"
    # check_C7_inf_C6 = None
    # if C7 <= C6 and C7 is not None and C6 is not None:
    #     check_C7_inf_C6 = "OK"
    # else:
    #     check_C7_inf_C6 = "NOK"
    # return
    
def test_BF_NASH_balanced_player_game_scenario1_setA1setB1setC1():
    fct_aux.N_DECIMALS = 8
    a, b = 1, 1
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 30 #0.33 #[15, 5]
    
    manual_debug = False
    gamma_version = -2  #-2,-1,1,2,3,4,5
    debug = False
    criteria_bf = "Perf_t"
    used_instances = True #False#True
    
    t_periods = 1
    
    path_2_arr_pl_setABC1 = os.path.join("tests", "AUTOMATE_INSTANCES_GAMES")
    path_file_2_arr_pl = os.path.join(path_2_arr_pl_setABC1,
                        "arr_pl_M_T_players_setA_1_setB_1_setC_1_periods_1_scenarioOnePeriod.npy")
    arr_pl_M_T_vars_init = np.load(path_file_2_arr_pl, allow_pickle=True)
    algo_names = fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    
    
    path_2_arr_pl_setABC_scen1 = os.path.join("tests","A1B1OnePeriod_50instances_ksteps100_b0.1_kstoplearn0.9",
                                         "OnePeriod_10instancesGammaV-2",
                                         "simu_DDMM_HHMM_6_t_1",
                                         "pi_hp_plus_10_pi_hp_minus_30",
                                         "BEST-BRUTE-FORCE")
    path_file_2_arr_pl = os.path.join(path_2_arr_pl_setABC_scen1,
                                      "arr_pl_M_T_K_vars_init.npy")
    arr_pl_M_T_vars_init = np.load(path_file_2_arr_pl, allow_pickle=True)
    algo_names = fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    
    
    global MOD
    m_players =arr_pl_M_T_vars_init.shape[0]
    MOD = int(0.10*pow(2, m_players)) \
            if pow(2, m_players) < 65000 \
            else int(0.020*pow(2, m_players))
    MOD = 1 if MOD == 0 else MOD
    
    name_simu = "BF_NASH_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    dico_profils_BF, dico_profils_NH, \
    dico_best_profils_BF, dico_bad_profils_BF, dico_mid_profils_BF, \
    dico_best_profils_NH, dico_bad_profils_NH, dico_mid_profils_NH \
       = BF_NASH_balanced_player_game(
            arr_pl_M_T_vars_init=arr_pl_M_T_vars_init.copy(), 
            algo_names=algo_names,
            pi_hp_plus=pi_hp_plus, 
            pi_hp_minus=pi_hp_minus,
            a=a, b=b,
            gamma_version=gamma_version,
            path_to_save=path_to_save, 
            name_dir="tests", 
            date_hhmm="DDMM_HHMM",
            manual_debug=manual_debug, 
            criteria_bf=criteria_bf, dbg=debug)
    
    
    C1 = True if dico_profils_NH["nb_profils"] > 0 else False
    C6s = dico_best_profils_BF["Perfs"] if dico_best_profils_BF["nb_best_profils"] > 0 else []
    C9s = dico_bad_profils_BF["Perfs"] if dico_bad_profils_BF["nb_bad_profils"] > 0 else []
    C7s_bad, C7s_best = [], []
    if C1:
       C7s_bad = dico_bad_profils_NH["Perfs"] if dico_bad_profils_NH["nb_bad_profils"] > 0 else []
       C7s_best = dico_best_profils_NH["Perfs"] if dico_best_profils_NH["nb_best_profils"] > 0 else []
       
    print("C1 = {}".format(C1))
    print("C6s={}, C9s={}".format(len(C6s), len(C9s) ))
    print("C7s_bad={}, C7s_best={}".format(len(C7s_bad), len(C7s_best) ))


def test_generer_balanced_players_4_modes_profils():
    setA_m_players, setB_m_players, setC_m_players = 1, 1, 1                   # 3 players
                      
    scenario_name = "scenarioOnePeriod"
    scenario = None
    
    name_dir = "tests"
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])    
    fct_aux.N_DECIMALS = 8
    a, b = 1, 1
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 30 #0.33 #[15, 5]
    
    manual_debug = False
    gamma_version = -2  #-2,-1,1,2,3,4,5
    debug = False
    criteria_bf = "Perf_t"
    used_instances = True #False#True
    
    pi_0_plus_t = fct_aux.PI_0_PLUS_INIT #4
    pi_0_minus_t = fct_aux.PI_0_MINUS_INIT #3
    
    t_periods = 1
    
    arr_pl_M_T_vars_init, arr_pl_MTvars_modif = None, None
    boolean_self = True 
    while boolean_self:
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc24(
                setA_m_players, setB_m_players, setC_m_players, 
                t_periods, 
                scenario,
                scenario_name,
                path_to_arr_pl_M_T, 
                used_instances)
        pi_hp_plus_T, pi_hp_minus_T, \
        phi_hp_plus_T, phi_hp_minus_T \
            = fct_aux.compute_pi_phi_HP_minus_plus_all_t(
                arr_pl_M_T_vars_init=arr_pl_M_T_vars_init,
                t_periods=t_periods,
                pi_hp_plus=pi_hp_plus,
                pi_hp_minus=pi_hp_minus,
                a=a,
                b=b, 
                gamma_version=gamma_version, 
                manual_debug=manual_debug,
                dbg=debug)
            
        t = 0
        
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
                                    dbg=debug)
        if "Self" in arr_pl_MTvars_modif[:,t, 
                                             fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]:
            boolean_self = False
        
    avant_state_Siold_Sis = arr_pl_MTvars_modif[:, t, 
                            [fct_aux.AUTOMATE_INDEX_ATTRS["state_i"], 
                             fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"], 
                             fct_aux.AUTOMATE_INDEX_ATTRS["Si"]
                             ]]
    
    m_players = arr_pl_MTvars_modif.shape[0]
    
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
    keys_best_BF_NH, keys_bad_BF_NH, \
    set_Perf_ts_BF, dico_modprofil_b0cO_Perf_t \
            = generer_balanced_players_4_modes_profils(
                arr_pl_MTvars_modif.copy(), 
                m_players, t,
                pi_hp_plus, pi_hp_minus,
                a, b,
                pi_0_plus_t, pi_0_minus_t,
                manual_debug, debug)
    
    apres_state_Siold_Sis = arr_pl_MTvars_modif[:, t, 
                            [fct_aux.AUTOMATE_INDEX_ATTRS["state_i"], 
                             fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"], 
                             fct_aux.AUTOMATE_INDEX_ATTRS["Si"]
                             ]]
    
    print("State_Siold_Si avant :{}, \n apres:{}".format(avant_state_Siold_Sis, 
                                                      apres_state_Siold_Sis))
    
    return arr_pl_MTvars_modif, list_dico_modes_profs_by_players_t_bestBF

###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    #test_BF_NASH_balanced_player_game()
    #test_BF_NASH_balanced_player_game_scenario1_setA1setB1setC1()
    arr, dico_bestBF = test_generer_balanced_players_4_modes_profils()
    
    print("runtime = {}".format(time.time() - ti))