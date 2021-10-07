import os
import sys
import time
import math
import json
import string
import random
import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players

from datetime import datetime
from pathlib import Path

#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
M_PLAYERS = 10
NUM_PERIODS = 50

CHOICE_RU = 1
N_DECIMALS = 2
NB_REPEAT_K_MAX = 4
STOP_LEARNING_PROBA = 0.90

Ci_LOW = 10
Ci_HIGH = 60

global PI_0_PLUS_INIT, PI_0_MINUS_INIT
PI_0_PLUS_INIT = 4 #20
PI_0_MINUS_INIT = 3 #10

SET_AC = ["setA", "setC"]
SET_ABC = ["setA", "setB", "setC"]
SET_AB1B2C = ["setA", "setB1",  "setB2", "setC"]
STATES = ["Deficit", "Self", "Surplus"]

STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("DIS", "PROD")

CASE1 = (1.7, 2.0) #(0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)

PROFIL_H = (0.6, 0.2, 0.2)
PROFIL_M = (0.2, 0.6, 0.2)
PROFIL_L = (0.2, 0.2, 0.6)

INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15}

NON_PLAYING_PLAYERS = {"PLAY":1, "NOT_PLAY":0}
ALGO_NAMES_BF = ["BEST-BRUTE-FORCE", "BAD-BRUTE-FORCE", "MIDDLE-BRUTE-FORCE"]
ALGO_NAMES_NASH = ["BEST-NASH", "BAD-NASH", "MIDDLE-NASH"]
ALGO_NAMES_DET = ["Selfish-DETERMINIST", "Systematic-DETERMINIST"]
ALGO_NAMES_LRIx = ["LRI1", "LRI2"]

# manual debug constants
MANUEL_DBG_GAMMA_I = np.random.randint(low=2, high=21, size=1)[0]              #5
MANUEL_DBG_PI_SG_PLUS_T_K = 8
MANUEL_DBG_PI_SG_MINUS_T_K = 10
MANUEL_DBG_PI_0_PLUS_T_K = 4 
MANUEL_DBG_PI_0_MINUS_T_K = 3

RACINE_PLAYER = "player"

name_cols_CX = {"C1":"C1", "C2":"C2", "C3":"C3", "C4":"C4", 
                "C5":"C5", "C6":"C6", "C7":"C7", "C9":"C9",
                "check_C5_inf_C6":"check_C5_inf_C6", 
                "check_C7_inf_C6":"check_C7_inf_C6", 
                "mean_proba_players":"mean_proba_players", 
                "nb_players_proba_inf_{}".format(STOP_LEARNING_PROBA):\
                    "nb_players_proba_inf_{}".format(STOP_LEARNING_PROBA),
                "players_proba_inf_{}".format(STOP_LEARNING_PROBA):\
                    "players_proba_inf_{}".format(STOP_LEARNING_PROBA)}

#_________________            AUTOMATE CONSTANCES           ________________

AUTOMATE_FILENAME_ARR_PLAYERS_ROOT = "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C = "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC = "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}_{}.npy"


AUTOMATE_INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, 
               "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15, "Si_minus":16,
               "Si_plus":17, "u_i": 18, "bg_i": 19,
               "S1_p_i_j_k": 20, "S2_p_i_j_k": 21, 
               "non_playing_players":22, "set":23}

#------------------------------------------------------------------------------
#                           definitions of class
#------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
#------------------------------------------------------------------------------
#                           definitions of functions
#------------------------------------------------------------------------------

###############################################################################
#                       fonctions transverses: debut       
###############################################################################
def fct_positive(sum_list1, sum_list2):
    """
    sum_list1 : sum of items in the list1
    sum_list2 : sum of items in the list2
    
    difference between sum of list1 et sum of list2 such as :
         diff = 0 if sum_list1 - sum_list2 <= 0
         diff = sum_list1 - sum_list2 if sum_list1 - sum_list2 > 0

        diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    Returns
    -------
    return 0 or sum_list1 - sum_list2
    
    """
    
    # boolean = sum_list1 - sum_list2 > 0
    # diff = boolean * (sum_list1 - sum_list2)
    diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    return diff

# _________        determine opposite mode_i: debut       _____________________
def find_out_opposite_mode(state_i, mode_i):
    """
    look for the opposite mode of the player.
    for example, 
    if state_i = Deficit, the possible modes are CONS+ and CONS-
    the opposite mode of CONS+ is CONS- and this of CONS- is CONS+
    """
    mode_i_bar = None
    if state_i == STATES[0] \
        and mode_i == STATE1_STRATS[0]:                                        # Deficit, CONS+
        mode_i_bar = STATE1_STRATS[1]                                          # CONS-
    elif state_i == STATES[0] \
        and mode_i == STATE1_STRATS[1]:                                        # Deficit, CONS-
        mode_i_bar = STATE1_STRATS[0]                                          # CONS+
    elif state_i == STATES[1] \
        and mode_i == STATE2_STRATS[0]:                                        # Self, DIS
        mode_i_bar = STATE2_STRATS[1]                                          # CONS-
    elif state_i == STATES[1] \
        and mode_i == STATE2_STRATS[1]:                                        # Self, CONS-
        mode_i_bar = STATE2_STRATS[0]                                          # DIS
    elif state_i == STATES[2] \
        and mode_i == STATE3_STRATS[0]:                                        # Surplus, DIS
        mode_i_bar = STATE3_STRATS[1]                                          # PROD
    elif state_i == STATES[2] \
        and mode_i == STATE3_STRATS[1]:                                        # Surplus, PROD
        mode_i_bar = STATE3_STRATS[0]                                          # DIS

    return mode_i_bar
# _________        determine opposite mode_i: fin         _____________________

def possibles_modes_players_automate(arr_pl_M_T_K_vars, t=0, k=0):
    """
    after remarking that some players have 2 states during the game, 
    I decide to write this function to set uniformly the players' state for all
    periods and all learning step

    Parameters
    ----------
    arr_pl_M_T_K_vars : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    m_players = arr_pl_M_T_K_vars.shape[0]
    possibles_modes = list()
    
    arr_pl_vars = None
    if len(arr_pl_M_T_K_vars.shape) == 3:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["state_i"]] 
            
            # get mode_i
            if state_i == STATES[0]:                                            # Deficit
                possibles_modes.append(STATE1_STRATS)
            elif state_i == STATES[1]:                                          # Self
                possibles_modes.append(STATE2_STRATS)
            elif state_i == STATES[2]:                                          # Surplus
                possibles_modes.append(STATE3_STRATS)
            # print("3: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, k, 
                                  AUTOMATE_INDEX_ATTRS["state_i"]]
            
            # get mode_i
            if state_i == STATES[0]:                                           # Deficit
                possibles_modes.append(STATE1_STRATS)
            elif state_i == STATES[1]:                                         # Self
                possibles_modes.append(STATE2_STRATS)
            elif state_i == STATES[2]:                                          # Surplus
                possibles_modes.append(STATE3_STRATS)
            # print("4: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
    else:
        print("STATE_i: NOTHING TO UPDATE.")
        
    return possibles_modes

###############################################################################
#                       fonctions transverses: fin       
###############################################################################


###############################################################################
#                       compute variables' values: debut       
###############################################################################

# _________         compute q_t_minus, q_t_plus:  fin          ________________

# _________       compute phi_hp_plus, phi_hp_minus: debut        _____________
def compute_cost_energy_bought_by_SG_2_HP(pi_hp_minus, quantity, b):
    """
    compute the cost of energy bought by SG to EPO
    """
    return pi_hp_minus * pow(quantity, b)

def compute_benefit_energy_sold_by_SG_2_HP(pi_hp_plus, quantity, a):
    """
    compute the benefit of energy sold by SG to EPO
    """
    return pi_hp_plus * pow(quantity, a)
# _________       compute phi_hp_plus, phi_hp_minus: fin          _____________

# _________         compute In_sg, Out_sg: debut          _____________________
def compute_prod_cons_SG(arr_pl_M_T, t):
    """
    compute the production In_sg and the consumption Out_sg in the SG.

    Parameters
    ----------
    arr_pl_M_T : array of shape (M_PLAYERS,NUM_PERIODS,len(INDEX_ATTRS))
        DESCRIPTION.
    t : integer
        DESCRIPTION.

    Returns
    -------
    In_sg, Out_sg : float, float.
    
    """
    In_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["prod_i"]].astype(np.float64) )
    Out_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["cons_i"]].astype(np.float64) )
    return In_sg, Out_sg
# _________         compute In_sg, Out_sg:  fin           _____________________

# _________         compute q_t_minus, q_t_plus:  debut        ________________
def compute_upper_bound_quantity_energy(arr_pl_M_T_K_vars_modif, t):
    """
    compute bought upper bound quantity energy q_t_minus
        and sold upper bound quantity energy q_t_plus
    """
    q_t_minus, q_t_plus = 0, 0
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi, Ci, Si, Si_max = None, None, None, None
        if len(arr_pl_M_T_K_vars_modif.shape) == 4:
            Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, AUTOMATE_INDEX_ATTRS["Si_max"]]
        else:
            Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]]
        diff_Ci_Pi = fct_positive(sum_list1=Ci, sum_list2=Pi)
        diff_Pi_Ci_Si_max = fct_positive(sum_list1=Pi, sum_list2=Ci+Si_max-Si)
        diff_Pi_Ci = fct_positive(sum_list1=Pi, sum_list2=Ci)
        diff_Ci_Pi_Si = fct_positive(sum_list1=Ci, sum_list2=Pi+Si)
        diff_Ci_Pi_Simax_Si = diff_Ci_Pi - diff_Pi_Ci_Si_max
        diff_Pi_Ci_Si = diff_Pi_Ci - diff_Ci_Pi_Si
        
        q_t_minus += diff_Ci_Pi_Simax_Si
        q_t_plus += diff_Pi_Ci_Si
        
        # print("Pi={}, Ci={}, Si_max={}, Si={}".format(Pi, Ci, Si_max, Si))
        # print("player {}: diff_Ci_Pi_Simax_Si={} -> q_t_minus={}, diff_Pi_Ci_Si={} -> q_t_plus={} ".format(
        #     num_pl_i, diff_Ci_Pi_Simax_Si, q_t_minus, diff_Pi_Ci_Si, q_t_plus))
        
    # print("q_t_minus={}, q_t_plus={}".format(q_t_minus, q_t_plus))
    q_t_minus = q_t_minus if q_t_minus >= 0 else 0
    q_t_plus = q_t_plus if q_t_plus >= 0 else 0
    return q_t_minus, q_t_plus
# _________         compute q_t_minus, q_t_plus:  fin          ________________

# _________             compute b0, c0: debut             _____________________
def compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                              pi_hp_plus, pi_hp_minus,
                              a, b,
                              In_sg, Out_sg):
    """
    compute the unit price of energy benefit and energy cost 
    
    pi_0_plus: the intern benefit price of one unit of energy inside SG
    pi_0_minus: the intern cost price of one unit of energy inside SG
    pi_hp_plus: the intern benefit price of one unit of energy between SG and HP
    pi_hp_minus: the intern cost price of one unit of energy between SG and HP
    Out_sg: the total amount of energy relative to the consumption of the SG
    In_sg: the total amount of energy relative to the production of the SG
    
    Returns
    -------
    bo: the benefit of one unit of energy in SG.
    co: the cost of one unit of energy in SG.

    """
    phi_hp_plus = compute_benefit_energy_sold_by_SG_2_HP(
                    pi_hp_plus=pi_hp_plus, 
                    quantity=In_sg-Out_sg, a=a)
    phi_hp_minus = compute_cost_energy_bought_by_SG_2_HP(
                    pi_hp_minus=pi_hp_minus, 
                    quantity=Out_sg-In_sg, b=b)
    c0 = pi_0_minus \
        if In_sg >= Out_sg \
        else (phi_hp_minus + In_sg*pi_0_minus)/Out_sg
    b0 = pi_0_plus \
        if In_sg < Out_sg \
        else (Out_sg * pi_0_plus + phi_hp_plus)/In_sg
   
    return round(b0, N_DECIMALS), round(c0, N_DECIMALS)
# _________             compute b0, c0: fin               _____________________

# _________          compute ben_i, cst_i: debut          _____________________
def compute_utility_players(arr_pl_M_T, gamma_is, t, b0, c0):
    """
    calculate the benefit and the cost of each player at time t

    Parameters
    ----------
    arr_pls_M_T : array of shape M_PLAYERS*NUM_PERIODS*9
        DESCRIPTION.
    gamma_is :  array of shape (M_PLAYERS,)
        DESCRIPTION.
    t : integer
        DESCRIPTION.
    b0 : float
        benefit per unit.
    c0 : float
        cost per unit.

    Returns
    -------
    bens: benefits of M_PLAYERS, shape (M_PLAYERS,).
    csts: costs of M_PLAYERS, shape (M_PLAYERS,)
    """
    bens = b0 * arr_pl_M_T[:, t, INDEX_ATTRS["prod_i"]] \
            + gamma_is * arr_pl_M_T[:, t, INDEX_ATTRS["r_i"]]
    csts = c0 * arr_pl_M_T[:, t, INDEX_ATTRS["cons_i"]]
    bens = np.around(np.array(bens, dtype=float), N_DECIMALS)
    csts = np.around(np.array(csts, dtype=float), N_DECIMALS)
    return bens, csts 
# _________          compute ben_i, cst_i: fin            _____________________

# _________    compute pi_sg_t_plus, pi_sg_t_minus: debut    __________________
def determine_new_pricing_sg(arr_pl_M_T, pi_hp_plus, pi_hp_minus, t, 
                             a, b, dbg=False):
    diff_energy_cons_t = 0
    diff_energy_prod_t = 0
    T = t+1
    for k in range(0, T):
        energ_k_prod = \
            fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]])
                    )
        energ_k_cons = \
            fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]])
                    )
            
        diff_energy_cons_t += energ_k_cons
        diff_energy_prod_t += energ_k_prod
        print("Price t={}, energ_k_prod={}, energ_k_cons={}".format(
            k, energ_k_prod, energ_k_cons)) if dbg else None
        bool_ = arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]>0
        unique,counts=np.unique(bool_,return_counts=True)
        sum_prod_k = round(np.sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]), 
                           N_DECIMALS)
        sum_cons_k = round(np.sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]]),
                           N_DECIMALS)
        diff_sum_prod_cons_k = sum_prod_k - sum_cons_k
        print("t={}, k={}, unique:{}, counts={}, sum_prod_k={}, sum_cons_k={}, diff_sum_k={}".format(
                t,k,unique, counts, sum_prod_k, sum_cons_k, diff_sum_prod_cons_k)) \
            if dbg==True else None
    
    sum_cons = sum(sum(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    sum_prod = sum(sum(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    print("NAN: cons={}, prod={}".format(
            np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)).any(),
            np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)).any())
        ) if dbg else None
    arr_cons = np.argwhere(np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    #arr_prod = np.argwhere(np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    if arr_cons.size != 0:
        for arr in arr_cons:
            print("{}-->state:{}, Pi={}, Ci={}, Si={}".format(
                arr, arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["state_i"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Pi"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Ci"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Si"]]))
                                
    phi_hp_minus = compute_cost_energy_bought_by_SG_2_HP(
                        pi_hp_minus=pi_hp_minus, 
                        quantity=diff_energy_cons_t, b=b)
    phi_hp_plus = compute_benefit_energy_sold_by_SG_2_HP(
                        pi_hp_plus=pi_hp_plus, 
                        quantity=diff_energy_prod_t, a=a)
    new_pi_sg_minus_t = round(phi_hp_minus / sum_cons, N_DECIMALS)  \
                    if sum_cons != 0 else np.nan
    new_pi_sg_plus_t = round(phi_hp_plus / sum_prod, N_DECIMALS) \
                        if sum_prod != 0 else np.nan
    # print("new_pi_sg_minus_t={}, new_pi_sg_plus_t={}, sum_cons={}, sum_prod={}, phi_hp_minus={}, phi_hp_plus={}"\
    #       .format(new_pi_sg_minus_t, new_pi_sg_plus_t, sum_cons, sum_prod, phi_hp_minus, phi_hp_plus))                    
                            
    return new_pi_sg_plus_t, new_pi_sg_minus_t

# _________    compute pi_sg_t_plus, pi_sg_t_minus: fin    ____________________

###############################################################################
#                       compute variables' values: fin       
###############################################################################

##############################################################################   
#                   compute gamma and state at each t ---> debut
##############################################################################
def get_values_Pi_Ci_Si_Simax_Pi1_Ci1(arr_pl_M_T_K_vars, 
                                      num_pl_i, t, k,
                                      t_periods,
                                      shape_arr_pl):
    """
    return the values of Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1 from arrays
    """
    Pi = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Pi"]]
    Ci = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Ci"]]
    Si_max = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Si_max"]]
    Si = None
    if t > 0:
        Si = arr_pl_M_T_K_vars[num_pl_i, t-1, 
                                       AUTOMATE_INDEX_ATTRS["Si"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t-1, k,
                                       AUTOMATE_INDEX_ATTRS["Si"]]
    else:
        Si = arr_pl_M_T_K_vars[num_pl_i, t, 
                                       AUTOMATE_INDEX_ATTRS["Si"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                       AUTOMATE_INDEX_ATTRS["Si"]]
    Ci_t_plus_1, Pi_t_plus_1 = None, None
    if t+1 > t_periods:
        Ci_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t+1, 
                                        AUTOMATE_INDEX_ATTRS["Ci"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t+1, k,
                                   AUTOMATE_INDEX_ATTRS["Ci"]]
        Pi_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t+1,
                                        AUTOMATE_INDEX_ATTRS["Pi"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t+1, k,
                                   AUTOMATE_INDEX_ATTRS["Pi"]]
    else:
        Ci_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["Ci"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                   AUTOMATE_INDEX_ATTRS["Ci"]]
        Pi_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS["Pi"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                   AUTOMATE_INDEX_ATTRS["Pi"]]
    
    return Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1

def update_variables(arr_pl_M_T_K_vars, variables, shape_arr_pl,
                     num_pl_i, t, k, gamma_i, Si,
                     pi_0_minus, pi_0_plus, 
                     pi_hp_minus_t, pi_hp_plus_t, dbg):
    
    # ____              update cell arrays: debut               _______
    if shape_arr_pl == 4:
        for (var,val) in variables:
            arr_pl_M_T_K_vars[num_pl_i, t, :,
                        AUTOMATE_INDEX_ATTRS[var]] = val
            
    elif shape_arr_pl == 3:
        for (var,val) in variables:
            arr_pl_M_T_K_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS[var]] = val
    # ____              update cell arrays: fin                 _______
    
    bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                    & (gamma_i <= max(pi_hp_minus_t, pi_hp_plus_t)+1)
    print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}"\
          .format(t, num_pl_i, gamma_i, bool_gamma_i)) if dbg else None

    Si_t_minus_1 = None
    if shape_arr_pl == 3 and t > 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t-1, 
                               AUTOMATE_INDEX_ATTRS["Si"]] 
    elif shape_arr_pl == 3 and t == 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t, 
                               AUTOMATE_INDEX_ATTRS["Si"]]
    elif shape_arr_pl == 4 and t > 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t-1, k, 
                               AUTOMATE_INDEX_ATTRS["Si"]] 
    elif shape_arr_pl == 4 and t == 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t, k,
                               AUTOMATE_INDEX_ATTRS["Si"]]
        
    print("Si_t_minus_1={}, Si={}".format(Si_t_minus_1, Si)) \
        if dbg else None
            
    return arr_pl_M_T_K_vars

def compute_gamma_state_4_period_t(arr_pl_M_T_K_vars, t, 
                                   pi_0_plus, pi_0_minus,
                                   pi_hp_plus_t, pi_hp_minus_t, 
                                   gamma_version=1, ppi_t_base=None, 
                                   manual_debug=False, dbg=False):
    """        
    compute gamma_i and state for all players 
    
    arr_pl_M_T_K_vars: shape (m_players, t_periods, len(vars)) or 
                             (m_players, t_periods, k_steps, len(vars))
    """
    m_players = arr_pl_M_T_K_vars.shape[0]
    t_periods = arr_pl_M_T_K_vars.shape[1]
    k = 0
    shape_arr_pl = len(arr_pl_M_T_K_vars.shape)
    
    Cis_t_plus_1 = arr_pl_M_T_K_vars[:, t, k, AUTOMATE_INDEX_ATTRS["Ci"]] \
                    if shape_arr_pl == 4 \
                    else arr_pl_M_T_K_vars[:, t, AUTOMATE_INDEX_ATTRS["Ci"]]
    Pis_t_plus_1 = arr_pl_M_T_K_vars[:, t, k, AUTOMATE_INDEX_ATTRS["Pi"]] \
                    if shape_arr_pl == 4 \
                    else arr_pl_M_T_K_vars[:, t, AUTOMATE_INDEX_ATTRS["Pi"]] 
    Cis_Pis_t_plus_1 = Cis_t_plus_1 - Pis_t_plus_1
    Cis_Pis_t_plus_1[Cis_Pis_t_plus_1 < 0] = 0
    GC_t = np.sum(Cis_Pis_t_plus_1)
    
    # initialisation of variables for gamma_version = 2
    state_is = np.empty(shape=(m_players,), dtype=object)
    Sis = np.zeros(shape=(m_players,))
    GSis_t_minus = np.zeros(shape=(m_players,)) 
    GSis_t_plus = np.zeros(shape=(m_players,))
    Xis = np.zeros(shape=(m_players,)) 
    Yis = np.zeros(shape=(m_players,))
    
    for num_pl_i in range(0, m_players):
        Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1 \
            = get_values_Pi_Ci_Si_Simax_Pi1_Ci1(
                arr_pl_M_T_K_vars, 
                num_pl_i, t, k,
                t_periods,
                shape_arr_pl)
        
        prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        state_i = pl_i.find_out_state_i()
        state_is[num_pl_i] = state_i
        Si_t_plus_1, Si_t_minus, Si_t_plus = None, None, None
        Xi, Yi, X_gamV5 = None, None, None
        if state_i == STATES[0]:                                               # Deficit or Deficit
            Si_t_minus = 0
            Si_t_plus = Si
            Xi = pi_0_minus
            Yi = pi_hp_minus_t
            X_gamV5 = pi_0_minus
        elif state_i == STATES[1]:                                             # Self or Self
            Si_t_minus = Si - (Ci - Pi)
            Si_t_plus = Si
            Xi = pi_0_minus
            Yi = pi_hp_minus_t
            X_gamV5 = pi_0_minus
        elif state_i == STATES[2]:                                             # Surplus or Surplus
            Si_t_minus = Si
            Si_t_plus = max(Si_max, Si+(Pi-Ci))
            Xi = pi_0_plus
            Yi = pi_hp_plus_t
            X_gamV5 = pi_0_plus
        Sis[num_pl_i] = Si
        GSis_t_minus[num_pl_i] = Si_t_minus
        GSis_t_plus[num_pl_i] = Si_t_plus
        Xis[num_pl_i] = Xi; Yis[num_pl_i] = Yi
        
        gamma_i, gamma_i_min, gamma_i_max, gamma_i_mid  = None, None, None, None
        res_mid = None
        ppi_t = None
        if manual_debug:
            gamma_i_min = MANUEL_DBG_GAMMA_I
            gamma_i_mid = MANUEL_DBG_GAMMA_I
            gamma_i_max = MANUEL_DBG_GAMMA_I
            gamma_i = MANUEL_DBG_GAMMA_I
        else:
            Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
            gamma_i_min = Xi - 1
            gamma_i_max = Yi + 1
            # print("Pi={}, Ci={}, Si={}, Si_t_plus_1={}, Si_t_minus={}, Si_t_plus={}".format(Pi, 
            #         Ci, Si, Si_t_plus_1, Si_t_minus, Si_t_plus))
            
            dif_pos_Ci_Pi_t_plus_1_Si_t_minus = 0
            dif_pos_Ci_Pi_t_plus_1_Si_t_minus = fct_positive(Si_t_plus_1, Si_t_minus)
            dif_Si_plus_minus = Si_t_plus - Si_t_minus
            frac = dif_pos_Ci_Pi_t_plus_1_Si_t_minus / dif_Si_plus_minus \
                    if dif_Si_plus_minus != 0 else 0
            ppi_t = np.sqrt(min(frac, 1))
            
            if Si_t_plus_1 < Si_t_minus:
                # Xi - 1
                gamma_i = gamma_i_min
            elif Si_t_plus_1 >= Si_t_plus:
                # Yi + 1
                gamma_i = gamma_i_max
            elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                        (Si_t_plus - Si_t_minus)
                Z = Xi + (Yi-Xi)*res_mid
                gamma_i_mid = int(np.floor(Z))
                gamma_i = gamma_i_mid
              
        # print("num_pl_i={},gamma_i={}, gamma_i_min={}, state_i={}, gamma_V{}".format(num_pl_i, 
        #         gamma_i, gamma_i_min, state_i, gamma_version))
        if gamma_version == 0:
            gamma_i = 0
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
            
        elif gamma_version == 1:
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
            
        elif gamma_version == 3:
            gamma_i = None
            if manual_debug:
                gamma_i = MANUEL_DBG_GAMMA_I
            elif Si_t_plus_1 < Si_t_minus:
                gamma_i = gamma_i_min
            else :
                gamma_i = gamma_i_max
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
            
        elif gamma_version == 4:
            gamma_i = None
            if manual_debug:
                gamma_i = MANUEL_DBG_GAMMA_I
            else:
                if Si_t_plus_1 < Si_t_minus:
                    # Xi - 1
                    gamma_i = gamma_i_min
                elif Si_t_plus_1 >= Si_t_plus:
                    # Yi + 1
                    gamma_i = gamma_i_max
                elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                    res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                            (Si_t_plus - Si_t_minus)
                    Z = Xi + (Yi-Xi) * np.sqrt(res_mid)
                    gamma_i_mid = int(np.floor(Z))
                    gamma_i = gamma_i_mid
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
            
        elif gamma_version == -1:
            gamma_i = np.random.randint(low=2, high=21, size=1)[0]
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
        
        elif gamma_version == 5:
            rd_draw = np.random.uniform(low=0.0, high=1.0, size=None)
            rho_i_t = 1 if rd_draw < ppi_t else 0
            gamma_i = rho_i_t * (X_gamV5 + 1)
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
        elif gamma_version == -2:
            ppi_t = 0.8 if ppi_t_base == None else ppi_t_base
            rd_draw = np.random.uniform(low=0.0, high=1.0, size=None)
            rho_i_t = 1 if rd_draw < ppi_t else 0
            gamma_i = rho_i_t * (X_gamV5 + 1)
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
            
        elif gamma_version == "NoNash":
            #gamma_i = pi_0_minus + 1 if num_pl_i == 0 else pi_0_plus + 1
            gamma_i = 3 if num_pl_i == 0 else 2
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
        elif gamma_version == "2Players10000Etapes":
            gamma_i = 3 if num_pl_i == 0 else 2
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus_t, pi_hp_plus_t, dbg)
        elif gamma_version == "instance6_10Players50000Etapes":
            arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["state_i"]] \
                = state_i
            
            
    if gamma_version == 2:
        GS_t_minus = np.sum(GSis_t_minus)
        GS_t_plus = np.sum(GSis_t_plus)
        gamma_is = None
        if GC_t <= GS_t_minus:
            gamma_is = Xis - 1
        elif GC_t > GS_t_plus:
            gamma_is = Yis + 1
        else:
            frac = (GC_t - GS_t_minus) / (GS_t_plus - GS_t_minus)
            res_is = Xis + (Yis-Xis)*frac
            gamma_is = np.floor(res_is)
            
        # ____              update cell arrays: debut               _______
        variables = [("Si", Sis), ("state_i", state_is), 
                     ("Si_minus", GSis_t_minus), ("Si_plus", GSis_t_plus)]
        if shape_arr_pl == 3:
            for (var,vals) in variables:
                arr_pl_M_T_K_vars[:, t,
                        AUTOMATE_INDEX_ATTRS[var]] = vals
            if manual_debug:
                arr_pl_M_T_K_vars[:, t,
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = MANUEL_DBG_GAMMA_I
            else:
                arr_pl_M_T_K_vars[:, t, 
                                  AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is
        elif shape_arr_pl == 4:
            # turn Sis 1D to 2D
            arrs_Sis_2D = []; arrs_state_is_2D = []; arrs_GSis_t_minus_2D = []
            arrs_GSis_t_plus_2D = []; arrs_gamma_is_2D = []; 
            arrs_manuel_dbg_gamma_i_2D = []; 
            manuel_dbg_gamma_is = [MANUEL_DBG_GAMMA_I for _ in range(0, m_players)]
            k_steps = arr_pl_M_T_K_vars.shape[2]
            for k in range(0, k_steps):
                arrs_Sis_2D.append(list(Sis))
                arrs_state_is_2D.append(list(state_is))
                arrs_GSis_t_minus_2D.append(list(GSis_t_minus))
                arrs_GSis_t_plus_2D.append(list(GSis_t_plus))
                arrs_gamma_is_2D.append(list(gamma_is))
                arrs_manuel_dbg_gamma_i_2D.append(manuel_dbg_gamma_is)
            Sis_2D = np.array(arrs_Sis_2D, dtype=object).T
            state_is_2D = np.array(arrs_state_is_2D, dtype=object).T
            GSis_t_minus_2D = np.array(arrs_GSis_t_minus_2D, dtype=object).T
            GSis_t_plus_2D = np.array(arrs_GSis_t_plus_2D, dtype=object).T
            gamma_is_2D = np.array(arrs_gamma_is_2D, dtype=object).T
            manuel_dbg_gamma_i_2D = np.array(arrs_manuel_dbg_gamma_i_2D, 
                                             dtype=object).T
            
            arr_pl_M_T_K_vars[:, t, :, AUTOMATE_INDEX_ATTRS["Si"]] = Sis_2D
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["state_i"]] = state_is_2D          
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_minus"]] = GSis_t_minus_2D
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_plus"]] = GSis_t_plus_2D 
            if manual_debug:
                arr_pl_M_T_K_vars[:, t, :,
                            AUTOMATE_INDEX_ATTRS["gamma_i"]] = manuel_dbg_gamma_i_2D
            else:
                arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is_2D
        
        # ____              update cell arrays: fin               _______
        
        bool_gamma_is = (gamma_is >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_is <= max(pi_hp_minus_t, pi_hp_plus_t)+1)
        print("GAMMA : t={}, val={}, bool_gamma_is={}"\
              .format(t, gamma_is, bool_gamma_is)) if dbg else None
        GSis_t_minus_1 = arr_pl_M_T_K_vars[
                            :, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                        if shape_arr_pl == 3 \
                        else arr_pl_M_T_K_vars[
                            :, t, k, AUTOMATE_INDEX_ATTRS["Si"]]
        print("GSis_t_minus_1={}, Sis={}".format(GSis_t_minus_1, Sis)) \
            if dbg else None
        
    
        
    
    return arr_pl_M_T_K_vars

# _____ compute gamma, pi_hp_{minus,plus}, phi_hp_{minus,plus}: debut   _______
def compute_pi_phi_HP_minus_plus_all_t(arr_pl_M_T_vars_init, t_periods,
                                             pi_hp_plus, pi_hp_minus,
                                             a, b, 
                                             gamma_version, 
                                             manual_debug, dbg):
    """
    compute, for all t, pi_hp_{plus, minus} and phi_hp_{plus, minus}
    
    return arr_pl_MTvars, pi_hp_plus_T, pi_hp_minus_T,
            phi_hp_plus_T, phi_hp_minus_T,
    """
    pi_hp_plus_T = np.empty(shape=(t_periods, )); pi_hp_plus_T.fill(np.nan)
    pi_hp_minus_T = np.empty(shape=(t_periods, )); pi_hp_minus_T.fill(np.nan)
    phi_hp_plus_T = np.empty(shape=(t_periods, )); phi_hp_plus_T.fill(np.nan)
    phi_hp_minus_T = np.empty(shape=(t_periods, )); phi_hp_minus_T.fill(np.nan)
    
    for t in range(0, t_periods):
        q_t_minus, q_t_plus = compute_upper_bound_quantity_energy(
                                    arr_pl_M_T_vars_init, t)
        phi_hp_minus_t = compute_cost_energy_bought_by_SG_2_HP(
                            pi_hp_minus=pi_hp_minus, 
                            quantity=q_t_minus,
                            b=b)
        phi_hp_plus_t = compute_benefit_energy_sold_by_SG_2_HP(
                            pi_hp_plus=pi_hp_plus, 
                            quantity=q_t_plus,
                            a=a)
        pi_hp_minus_t = round(phi_hp_minus_t/q_t_minus, N_DECIMALS) \
                        if q_t_minus != 0 \
                        else 0
        pi_hp_plus_t = round(phi_hp_plus_t/q_t_plus, N_DECIMALS) \
                        if q_t_plus != 0 \
                        else 0
                        
        pi_hp_plus_T[t] = pi_hp_plus_t
        pi_hp_minus_T[t] = pi_hp_minus_t
        phi_hp_plus_T[t] = phi_hp_plus_t
        phi_hp_minus_T[t] = phi_hp_minus_t
        
        
    return pi_hp_plus_T, pi_hp_minus_T, phi_hp_plus_T, phi_hp_minus_T
                        
# _____ compute gamma, pi_hp_{minus,plus}, phi_hp_{minus,plus}: fin     _______

##############################################################################   
#                   compute gamma and state at each t ---> fin
##############################################################################


###############################################################################
#           balanced players 4 mode profile at t and k --> debut   
###############################################################################
# _________    compute prices for not-learning algos: debut   _________________
def compute_prices_inside_SG_4_notLearnAlgo(arr_pl_M_T_vars_modif, t,
                                            pi_hp_plus, pi_hp_minus,
                                            a, b,
                                            pi_0_plus_t, pi_0_minus_t,
                                            manual_debug, dbg):
    
    # compute the new prices pi_sg_plus_t, pi_sg_minus_t
    # from a pricing model in the document
    pi_sg_plus_t, pi_sg_minus_t = determine_new_pricing_sg(
                                            arr_pl_M_T_vars_modif, 
                                            pi_hp_plus, pi_hp_minus, 
                                            a=a, b=b,
                                            t=t, dbg=dbg)
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = compute_prod_cons_SG(arr_pl_M_T_vars_modif, t)
    # print("In_sg={}, Out_sg={}".format(In_sg, Out_sg ))
    
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    a=a, b=b,
                    In_sg=In_sg, Out_sg=Out_sg)
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars_modif[:, t, AUTOMATE_INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = compute_utility_players(arr_pl_M_T_vars_modif, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    
    return b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t
# _________     compute prices for not-learning algos: fin    _________________

def balanced_player_game_4_mode_profil(arr_pl_M_T_vars_modif, 
                                        mode_profile,
                                        t,
                                        manual_debug, dbg):
    
    dico_gamma_players_t = dict()
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars_modif[num_pl_i, t,
                                   AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars_modif[num_pl_i, t,
                                   AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   AUTOMATE_INDEX_ATTRS['Si']] 
        Si_max = arr_pl_M_T_vars_modif[num_pl_i, t,
                                       AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i = 0, 0, 0
        state_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                            # update R_i_old
                
        # select mode for player num_pl_i
        mode_i = mode_profile[num_pl_i]
        pl_i.set_mode_i(mode_i)
        
        # compute cons, prod, r_i
        pl_i.update_prod_cons_r_i()
        
        # is pl_i balanced?
        boolean, formule = balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", pl_i.get_mode_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_M_T_vars_modif[num_pl_i, t,
                                    AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars_modif, dico_gamma_players_t

def balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                            arr_pl_M_T_vars_modif, 
                            mode_profile,
                            t,
                            pi_hp_plus, pi_hp_minus, 
                            a, b,
                            pi_0_plus_t, pi_0_minus_t,
                            random_mode,
                            manual_debug, dbg=False):
    """
    """
    # find mode, prod, cons, r_i
    arr_pl_M_T_vars_modif, dico_gamma_players_t \
        = balanced_player_game_4_mode_profil(
            arr_pl_M_T_vars_modif.copy(), 
            mode_profile,
            t,
            manual_debug, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t, \
        = compute_prices_inside_SG_4_notLearnAlgo(arr_pl_M_T_vars_modif, t,
                                                   pi_hp_plus, pi_hp_minus,
                                                   a, b,
                                                   pi_0_plus_t, pi_0_minus_t,
                                                   manual_debug, dbg)
        
    return arr_pl_M_T_vars_modif, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamma_players_t

###############################################################################
#           balanced players 4 mode profile at t and k --> fin   
###############################################################################

###############################################################################
#                    checkout LRI profil --> debut 
###############################################################################
def checkout_nash_4_profils_by_periods(arr_pl_M_T_vars_modif,
                                        arr_pl_M_T_vars,
                                        pi_hp_plus, pi_hp_minus, 
                                        a, b,
                                        pi_0_minus_t, pi_0_plus_t, 
                                        bens_csts_M_t,
                                        t,
                                        manual_debug):
    """
    verify if the profil at time t and k_stop_learning is a Nash equilibrium.
    """
    # create a result dataframe of checking players' stability and nash equilibrium
    cols = ["players", "nash_modes_t{}".format(t), 'states_t{}'.format(t), 
            'Vis_t{}'.format(t), 'Vis_bar_t{}'.format(t), 
               'res_t{}'.format(t)] 
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    id_players = list(range(0, m_players))
    df_nash_t = pd.DataFrame(index=id_players, columns=cols)
    
    # revert Si to the initial value ie at t and k=0
    Sis = arr_pl_M_T_vars[:, t,
                          AUTOMATE_INDEX_ATTRS["Si"]]
    arr_pl_M_T_vars_modif[:, t,
                          AUTOMATE_INDEX_ATTRS["Si"]] = Sis
    
    # stability of each player
    modes_profil = list(arr_pl_M_T_vars_modif[
                            :, t,
                            AUTOMATE_INDEX_ATTRS["mode_i"]] )
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_M_T_vars_modif[
                        num_pl_i, t,
                        AUTOMATE_INDEX_ATTRS["state_i"]] 
        mode_i = modes_profil[num_pl_i]
        mode_i_bar = find_out_opposite_mode(state_i, mode_i)
        
        opposite_modes_profil = modes_profil.copy()
        opposite_modes_profil[num_pl_i] = mode_i_bar
        opposite_modes_profil = tuple(opposite_modes_profil)
        
        df_nash_t.loc[num_pl_i, "players"] = RACINE_PLAYER+"_"+str(num_pl_i)
        df_nash_t.loc[num_pl_i, "nash_modes_t{}".format(t)] = mode_i
        df_nash_t.loc[num_pl_i, "states_t{}".format(t)] = state_i
        
        random_mode = False
        arr_pl_M_T_K_vars_modif_mode_prof_BAR, \
        b0_t_k_bar, c0_t_k_bar, \
        bens_t_k_bar, csts_t_k_bar, \
        pi_sg_plus_t_k_bar, pi_sg_minus_t_k_bar, \
        dico_gamma_players_t_k \
            = balanced_player_game_t_4_mode_profil_prices_SG_4_notLearnAlgo(
                    arr_pl_M_T_vars_modif.copy(), 
                    opposite_modes_profil,
                    t, 
                    pi_hp_plus, pi_hp_minus, 
                    a, b,
                    pi_0_plus_t, pi_0_minus_t,
                    random_mode,
                    manual_debug, dbg=False)
        
        Vi = bens_csts_M_t[num_pl_i]
        bens_csts_t_k_bar = bens_t_k_bar - csts_t_k_bar
        Vi_bar = bens_csts_t_k_bar[num_pl_i]
    
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
###############################################################################
#                       checkout LRI profil --> fin 
###############################################################################

###############################################################################
#              compute prices B C BB CC RU ---> debut                
###############################################################################
# _____             compute prices B C BB CC RU ---> fin                 _____
def compute_prices_B_C_BB_CC_EB_DET(arr_pl_M_T_vars_modif, 
                                    pi_sg_minus_T, pi_sg_plus_T, 
                                    pi_0_minus_T, pi_0_plus_T,
                                    b0_s_T, c0_s_T):
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    
    B_is_M = np.empty(shape=(m_players, )); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players, )); C_is_M.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players, )); BB_is_M.fill(np.nan)
    CC_is_M = np.empty(shape=(m_players, )); CC_is_M.fill(np.nan)
    EB_is_M = np.empty(shape=(m_players, )); EB_is_M.fill(np.nan)
    
    B_is_MT = np.empty(shape=(m_players, t_periods)); B_is_MT.fill(np.nan)
    C_is_MT = np.empty(shape=(m_players, t_periods)); C_is_MT.fill(np.nan)
    B_is_MT_cum = np.empty(shape=(m_players, t_periods)); B_is_MT_cum.fill(np.nan) # cum = cumul of B from 0 to t-1 
    C_is_MT_cum = np.empty(shape=(m_players, t_periods)); C_is_MT_cum.fill(np.nan)
    BB_is_MT = np.empty(shape=(m_players, t_periods)); BB_is_MT.fill(np.nan)
    CC_is_MT = np.empty(shape=(m_players, t_periods)); CC_is_MT.fill(np.nan)
    EB_is_MT = np.empty(shape=(m_players, t_periods)); EB_is_MT.fill(np.nan)
    
    PROD_is_MT = np.empty(shape=(m_players, t_periods)); PROD_is_MT.fill(np.nan)
    CONS_is_MT = np.empty(shape=(m_players, t_periods)); CONS_is_MT.fill(np.nan)
    PROD_is_MT_cum = np.empty(shape=(m_players, t_periods)); PROD_is_MT_cum.fill(np.nan)
    CONS_is_MT_cum = np.empty(shape=(m_players, t_periods)); CONS_is_MT_cum.fill(np.nan)
    for t in range(0, t_periods):
        prod_is_Mt = arr_pl_M_T_vars_modif[:,t, AUTOMATE_INDEX_ATTRS["prod_i"]]
        cons_is_Mt = arr_pl_M_T_vars_modif[:,t, AUTOMATE_INDEX_ATTRS["cons_i"]]
        b0_t = b0_s_T[t]; c0_t = c0_s_T[t]
        B_is_MT[:,t] = b0_t * prod_is_Mt
        C_is_MT[:,t] = c0_t * cons_is_Mt
        B_is_MT_cum[:,t] = np.sum(B_is_MT[:,:t+1], axis=1) 
        C_is_MT_cum[:,t] = np.sum(C_is_MT[:,:t+1], axis=1) 
        
        PROD_is_MT[:,t] = prod_is_Mt
        CONS_is_MT[:,t] = cons_is_Mt
        CC_is_MT[:,t] = pi_sg_minus_T[t] * np.sum(CONS_is_MT[:,:t+1], axis=1) 
        BB_is_MT[:,t] = pi_sg_plus_T[t] * np.sum(PROD_is_MT[:,:t+1], axis=1) 
        EB_is_MT[:,t] = BB_is_MT[:,t] - CC_is_MT[:,t]
        
    B_is_M = np.sum(B_is_MT[:, :], axis=1)
    C_is_M = np.sum(C_is_MT[:, :], axis=1)
    BB_is_M = BB_is_MT[:, t_periods-1]
    CC_is_M = CC_is_MT[:, t_periods-1]
    EB_is_M = EB_is_MT[:, t_periods-1]
    
    return B_is_M, C_is_M, BB_is_M, CC_is_M, EB_is_M, \
           B_is_MT_cum, C_is_MT_cum, BB_is_MT, CC_is_MT, EB_is_MT, \
           B_is_MT, C_is_MT 
# _____             compute prices B C BB CC RU ---> fin                 _____

# _____         checkout prices from computing variables ---> debut      _____
def checkout_prices_4_computing_variables_DET(arr_pl_M_T_vars_modif, 
                                          pi_sg_minus_T, pi_sg_plus_T, 
                                          pi_0_minus_T, pi_0_plus_T,
                                          b0_s_T, c0_s_T,
                                          B_is_M, C_is_M ,
                                          BB_is_M, CC_is_M, EB_is_M):
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    
    B_M_cp = np.empty(shape=(m_players, )); B_M_cp.fill(np.nan)
    C_M_cp = np.empty(shape=(m_players, )); C_M_cp.fill(np.nan)
    BB_M_cp = np.empty(shape=(m_players, )); BB_M_cp.fill(np.nan)
    CC_M_cp = np.empty(shape=(m_players, )); CC_M_cp.fill(np.nan)
    EB_M_cp = np.empty(shape=(m_players, )); EB_M_cp.fill(np.nan)
    
    B_MT_cp = np.empty(shape=(m_players, t_periods)); B_MT_cp.fill(np.nan)
    C_MT_cp = np.empty(shape=(m_players, t_periods)); C_MT_cp.fill(np.nan)
    B_MT_cum_cp = np.empty(shape=(m_players, t_periods)); B_MT_cum_cp.fill(np.nan) # cum = cumul of B from 0 to t-1 
    C_MT_cum_cp = np.empty(shape=(m_players, t_periods)); C_MT_cum_cp.fill(np.nan)
    BB_MT_cp = np.empty(shape=(m_players, t_periods)); BB_MT_cp.fill(np.nan)
    CC_MT_cp = np.empty(shape=(m_players, t_periods)); CC_MT_cp.fill(np.nan)
    EB_MT_cp = np.empty(shape=(m_players, t_periods)); EB_MT_cp.fill(np.nan)
    
    PROD_MT = np.empty(shape=(m_players, t_periods)); PROD_MT.fill(np.nan)
    CONS_MT = np.empty(shape=(m_players, t_periods)); CONS_MT.fill(np.nan)
    PROD_MT_cum = np.empty(shape=(m_players, t_periods)); PROD_MT_cum.fill(np.nan)
    CONS_MT_cum = np.empty(shape=(m_players, t_periods)); CONS_MT_cum.fill(np.nan)
    for t in range(0, t_periods):
        prod_is_Mt = arr_pl_M_T_vars_modif[:,t, AUTOMATE_INDEX_ATTRS["prod_i"]]
        cons_is_Mt = arr_pl_M_T_vars_modif[:,t, AUTOMATE_INDEX_ATTRS["cons_i"]]
        b0_t = b0_s_T[t]; c0_t = c0_s_T[t]
        B_MT_cp[:,t] = b0_t * prod_is_Mt
        C_MT_cp[:,t] = c0_t * cons_is_Mt
        B_MT_cum_cp[:,t] = np.sum(B_MT_cp[:,:t+1], axis=1)
        C_MT_cum_cp[:,t] = np.sum(C_MT_cp[:,:t+1], axis=1) 
        
        PROD_MT[:,t] = prod_is_Mt
        CONS_MT[:,t] = cons_is_Mt
        CC_MT_cp[:,t] = pi_sg_minus_T[t] * np.sum(CONS_MT[:,:t+1], axis=1)
        BB_MT_cp[:,t] = pi_sg_plus_T[t] * np.sum(PROD_MT[:,:t+1], axis=1)
        EB_MT_cp[:,t] = BB_MT_cp[:,t] - CC_MT_cp[:,t]  
        
    B_M_cp = np.sum(B_MT_cp[:, :], axis=1)
    C_M_cp = np.sum(C_MT_cp[:, :], axis=1)
    BB_M_cp = BB_MT_cp[:, t_periods-1]
    CC_M_cp = CC_MT_cp[:, t_periods-1]
    EB_M_cp = EB_MT_cp[:, t_periods-1]
    
    cpt_BB_OK, cpt_CC_OK = 0, 0
    for num_pli in range(0, m_players):
        if np.abs(BB_M_cp[num_pli] - BB_is_M[num_pli]) < pow(10,-1):
            cpt_BB_OK += 1
        if np.abs(CC_M_cp[num_pli] - CC_is_M[num_pli]) < pow(10,-1):
            cpt_CC_OK += 1
            
    print("BB_M OK?: {}, CC_M OK?:{}".format(round(cpt_BB_OK/m_players, 2), 
                                             round(cpt_CC_OK/m_players, 2)))
        
def checkout_prices_B_C_BB_CC_EB_DET(arr_pl_M_T_vars_modif, 
                                     path_to_save):
    
    print("path_to_save={}".format(path_to_save) )
    
    # read from hard disk
    arr_pl_M_T_vars, \
    b0_s_T, c0_s_T, \
    B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
    BENs_M_T_K, CSTs_M_T_K, \
    BB_is_M, CC_is_M, EB_is_M, BB_is_M_T, CC_is_M_T, EB_is_M_T,\
    pi_sg_plus_T, pi_sg_minus_T, \
    pi_0_plus_T, pi_0_minus_T, \
    pi_hp_plus_T, pi_hp_minus_T \
        = get_local_storage_variables(path_to_variable=path_to_save)
        
    checkout_prices_4_computing_variables_DET(arr_pl_M_T_vars, 
                                          pi_sg_minus_T = pi_sg_minus_T, 
                                          pi_sg_plus_T = pi_sg_plus_T, 
                                          pi_0_minus_T = pi_0_minus_T, 
                                          pi_0_plus_T = pi_0_plus_T,
                                          b0_s_T = b0_s_T, c0_s_T = c0_s_T,
                                          B_is_M = B_is_M, C_is_M = C_is_M,
                                          BB_is_M = BB_is_M, CC_is_M = CC_is_M, 
                                          EB_is_M = EB_is_M)

###############################################################################
#              checkout prices from computing variables ---> fin        
###############################################################################

###############################################################################
#                       save variables: debut       
###############################################################################
def save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K,
                   B_is_M, C_is_M, B_is_M_T, C_is_M_T,
                   BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, EB_is_M, 
                   BB_is_M_T, CC_is_M_T, EB_is_M_T,
                   dico_EB_R_EBsetA1B1_EBsetB2C,
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_T, pi_hp_minus_T, dico_stats_res,
                   algo="LRI",
                   dico_best_steps=dict()):
    
    if algo is None:
        path_to_save = path_to_save \
                        if path_to_save != "tests" \
                        else os.path.join(
                                    path_to_save, 
                                    "simu_"+datetime.now()\
                                        .strftime("%d%m_%H%M"))
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
    else:
        path_to_save = path_to_save \
                        if path_to_save != "tests" \
                        else os.path.join(
                                    path_to_save, 
                                    algo+"_simu_"+datetime.now()\
                                        .strftime("%d%m_%H%M"))
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
        
    np.save(os.path.join(path_to_save, "arr_pl_M_T_K_vars.npy"), 
            arr_pl_M_T_K_vars)
    np.save(os.path.join(path_to_save, "b0_s_T_K.npy"), b0_s_T_K)
    np.save(os.path.join(path_to_save, "c0_s_T_K.npy"), c0_s_T_K)
    np.save(os.path.join(path_to_save, "B_is_M.npy"), B_is_M)
    np.save(os.path.join(path_to_save, "C_is_M.npy"), C_is_M)
    
    np.save(os.path.join(path_to_save, "B_is_M_T.npy"), B_is_M_T)
    np.save(os.path.join(path_to_save, "C_is_M_T.npy"), C_is_M_T)
    
    np.save(os.path.join(path_to_save, "BENs_M_T_K.npy"), BENs_M_T_K)
    np.save(os.path.join(path_to_save, "CSTs_M_T_K.npy"), CSTs_M_T_K)
    np.save(os.path.join(path_to_save, "BB_is_M.npy"), BB_is_M)
    np.save(os.path.join(path_to_save, "CC_is_M.npy"), CC_is_M)
    np.save(os.path.join(path_to_save, "EB_is_M.npy"), EB_is_M)
    
    np.save(os.path.join(path_to_save, "BB_is_M_T.npy"), BB_is_M_T)
    np.save(os.path.join(path_to_save, "CC_is_M_T.npy"), CC_is_M_T)
    np.save(os.path.join(path_to_save, "EB_is_M_T.npy"), EB_is_M_T)
    
    np.save(os.path.join(path_to_save, "pi_sg_minus_T_K.npy"), pi_sg_minus_T_K)
    np.save(os.path.join(path_to_save, "pi_sg_plus_T_K.npy"), pi_sg_plus_T_K)
    np.save(os.path.join(path_to_save, "pi_0_minus_T_K.npy"), pi_0_minus_T_K)
    np.save(os.path.join(path_to_save, "pi_0_plus_T_K.npy"), pi_0_plus_T_K)
    np.save(os.path.join(path_to_save, "pi_hp_plus_T.npy"), pi_hp_plus_T)
    np.save(os.path.join(path_to_save, "pi_hp_minus_T.npy"), pi_hp_minus_T)
    pd.DataFrame.from_dict(dico_stats_res)\
        .to_csv(os.path.join(path_to_save, "stats_res.csv"))
    pd.DataFrame.from_dict(dico_best_steps)\
        .to_csv(os.path.join(path_to_save,
                              "best_learning_steps.csv"), 
                  index=True)
    pd.DataFrame.from_dict(dico_best_steps, orient="columns")\
        .to_excel(os.path.join(path_to_save,
                               "best_learning_steps.xlsx"), 
                  index=True)
    pd.DataFrame(dico_EB_R_EBsetA1B1_EBsetB2C, index=["values"]).T\
        .to_excel(os.path.join(path_to_save,
                               "EB_R_EBsetA1B1_EBsetB2C.xlsx"), 
                  index=True)
        
    
    print("$$$$ saved variables. $$$$")
    
def save_instances_games(arr_pl_M_T, name_file_arr_pl, path_to_save):
    """
    Store players instances of the game so that:
        the players' numbers, the periods' numbers, the scenario and the prob_Ci
        define a game.
        
    Parameters:
        ----------
    arr_pl_M_T : array of shape (M_PLAYERS,NUM_PERIODS,len(INDEX_ATTRS))
        DESCRIPTION.
    name_file_arr_pl : string 
        DESCRIPTION.
        Name of file saving arr_pl_M_T
    path_to_save : string
        DESCRIPTION.
        path to save instances
        
    """
    
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(path_to_save, name_file_arr_pl), 
            arr_pl_M_T)
    
###############################################################################
#                           save variables: fin       
###############################################################################

############################################################################### 
#               
#        get local variables and turn them into dataframe --> debut
###############################################################################
def get_local_storage_variables(path_to_variable):
    """
    obtain the content of variables stored locally .

    Returns
    -------
     arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s.
    
    arr_pls_M_T: array of players with a shape M_PLAYERS*T_PERIODS*INDEX_ATTRS
    arr_T_nsteps_vars : array of players with a shape 
                        M_PLAYERS*T_PERIODS*NSTEPS*vars_nstep
                        avec len(vars_nstep)=20
    RUs: array of (M_PLAYERS,)
    BENs: array of M_PLAYERS*T_PERIODS
    CSTs: array of M_PLAYERS*T_PERIODS
    B0s: array of (T_PERIODS,)
    C0s: array of (T_PERIODS,)
    pi_sg_plus_s: array of (T_PERIODS,)
    pi_sg_minus_s: array of (T_PERIODS,)

    pi_hp_plus_s: array of (T_PERIODS,)
    pi_hp_minus_s: array of (T_PERIODS,)
    """

    arr_pl_M_T_K_vars = np.load(os.path.join(path_to_variable, 
                                             "arr_pl_M_T_K_vars.npy"),
                          allow_pickle=True)
    b0_s_T_K = np.load(os.path.join(path_to_variable, "b0_s_T_K.npy"),
                          allow_pickle=True)
    c0_s_T_K = np.load(os.path.join(path_to_variable, "c0_s_T_K.npy"),
                          allow_pickle=True)
    B_is_M = np.load(os.path.join(path_to_variable, "B_is_M.npy"),
                          allow_pickle=True)
    C_is_M = np.load(os.path.join(path_to_variable, "C_is_M.npy"),
                          allow_pickle=True)
    B_is_M_T = np.load(os.path.join(path_to_variable, "B_is_M_T.npy"),
                          allow_pickle=True)
    C_is_M_T = np.load(os.path.join(path_to_variable, "C_is_M_T.npy"),
                          allow_pickle=True)
    BENs_M_T_K = np.load(os.path.join(path_to_variable, "BENs_M_T_K.npy"),
                          allow_pickle=True)
    CSTs_M_T_K = np.load(os.path.join(path_to_variable, "CSTs_M_T_K.npy"),
                          allow_pickle=True)
    BB_is_M = np.load(os.path.join(path_to_variable, "BB_is_M.npy"),
                          allow_pickle=True)
    CC_is_M = np.load(os.path.join(path_to_variable, "CC_is_M.npy"),
                          allow_pickle=True)
    EB_is_M = np.load(os.path.join(path_to_variable, "EB_is_M.npy"),
                          allow_pickle=True)
    BB_is_M_T = np.load(os.path.join(path_to_variable, "BB_is_M_T.npy"),
                          allow_pickle=True)
    CC_is_M_T = np.load(os.path.join(path_to_variable, "CC_is_M_T.npy"),
                          allow_pickle=True)
    EB_is_M_T = np.load(os.path.join(path_to_variable, "EB_is_M_T.npy"),
                          allow_pickle=True)
    pi_sg_plus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_plus_T_K.npy"),
                          allow_pickle=True)
    pi_sg_minus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_minus_T_K.npy"),
                          allow_pickle=True)
    pi_0_plus_T_K = np.load(os.path.join(path_to_variable, "pi_0_plus_T_K.npy"),
                          allow_pickle=True)
    pi_0_minus_T_K = np.load(os.path.join(path_to_variable, "pi_0_minus_T_K.npy"),
                          allow_pickle=True)
    pi_hp_plus_T = np.load(os.path.join(path_to_variable, "pi_hp_plus_T.npy"),
                          allow_pickle=True)
    pi_hp_minus_T = np.load(os.path.join(path_to_variable, "pi_hp_minus_T.npy"),
                          allow_pickle=True)
    
    return arr_pl_M_T_K_vars, \
            b0_s_T_K, c0_s_T_K, \
            B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
            BENs_M_T_K, CSTs_M_T_K, \
            BB_is_M, CC_is_M, EB_is_M, BB_is_M_T, CC_is_M_T, EB_is_M_T,\
            pi_sg_plus_T_K, pi_sg_minus_T_K, \
            pi_0_plus_T_K, pi_0_minus_T_K, \
            pi_hp_plus_T, pi_hp_minus_T
            
############################################################################### 
#               
#        get local variables and turn them into dataframe --> fin
################################################################################

###############################################################################
#            generate Pi, Ci by automate --> debut
###############################################################################
def generate_Pi_Ci_one_period(setA_m_players=15, 
                               setB_m_players=10, 
                               setC_m_players=10, 
                               t_periods=1, 
                               scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA belonging to Deficit
             setB_m_players = number of players in setB belonging to Self
             setC_m_players = number of players in setC belonging to Surplus
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            prob = np.random.random(size=1)[0]
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_ABC[0]:                                             # setA = State1 or Deficit
                Si_t = 0
                Ci_t = 15
                Pi_t = np.random.randint(low=5, high=10, size=1)[0]
                
            elif setX == SET_ABC[1] and prob<0.5:                              # setB1 = State2 or Self
                Si_t = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=5, high=8, size=1)[0]
                
            elif setX == SET_ABC[1] and prob>=0.5:                              # setB1 State2 or Self
                Si_t = 8
                Ci_t = 31
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]
                
            elif setX == SET_ABC[2]:                                           # setC State3 or Surplus
                Si_t = 8
                Ci_t = 20
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB_m_players = number of players in setB
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    (prob_A_A, prob_A_B, prob_A_C) = scenario[0]
    (prob_B_A, prob_B_B, prob_B_C) = scenario[1]
    (prob_C_A, prob_C_B, prob_C_C) = scenario[2]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_ABC[0]:                                             # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=8, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_A_A, 
                                                              prob_A_B, 
                                                              prob_A_C])
            elif setX == SET_ABC[1]:                                           # setB
                Si_t = 7
                Ci_t = 20
                x = np.random.randint(low=12, high=20, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_B_A, 
                                                              prob_B_B, 
                                                              prob_B_C])
            elif setX == SET_ABC[2]:                                           # setC
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=26, high=35, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_C_A, 
                                                              prob_C_B, 
                                                              prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [2,8]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[1]:                                           # setB
                Pis = [12,20]; Cis = [20]; Sis = [4]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [26,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [5,10]; Cis = [15]; Sis = [0]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_ABC[1]:                                           # setB
                Pis_1 = [5,8]; Cis_1 = [10]; Sis_1 = [6]
                Pis_2 = [21,30]; Cis_2 = [31]; Sis_2 = [8]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if (Pi >= Pis_1[0] and Pi <= Pis_1[1]) \
                                    or (Pi >= Pis_2[0] and Pi <= Pis_2[1]) else 0
                cpt_t_Pi_nok += 1 if (Pi < Pis_1[0] or Pi > Pis_1[1]) \
                                     and (Pi < Pis_2[0] or Pi > Pis_2[1]) else 0
                cpt_t_Ci_ok += 1 if (Ci in Cis_1 or Ci in Cis_2)  else 0
                cpt_t_Ci_nok += 1 if (Ci not in Cis_1 and Ci not in Cis_2) else 0
                cpt_t_Si_ok += 1 if Si in Sis_1 or Si in Sis_2 else 0
                cpt_t_Si_nok += 1 if Si not in Sis_1 and Si not in Sis_2 else 0
                
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [21,30]; Cis = [20]; Sis = [8]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate --> fin
###############################################################################

###############################################################################
#                   generate Pi, Ci One Period docNoNASH--> debut
###############################################################################
def generate_Pi_Ci_one_period_docNONASH(setA_m_players, 
                                        setB_m_players, 
                                        setC_m_players, 
                                        t_periods, 
                                        scenario):
    setA_id_players = [0]
    setC_id_players = [1]
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    # player1
    num_pl_i = 0
    Si_t = 1
    Si_t_max = 2
    Ci_t = 3 #10
    Pi_t = 1 #8
    # update arrays cells with variables
    cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
            ("mode_i",""), ("state_i","")]
    for col, val in cols:
        arr_pl_M_T_vars[num_pl_i, t, 
                        AUTOMATE_INDEX_ATTRS[col]] = val
        
    # player2
    num_pl_i = 1
    Si_t = 1
    Si_t_max = 2 #Si_t + 1
    Ci_t = 1 #8
    Pi_t = 3 #10
    # update arrays cells with variables
    cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
            ("mode_i",""), ("state_i","")]
    for col, val in cols:
        arr_pl_M_T_vars[num_pl_i, t, 
                        AUTOMATE_INDEX_ATTRS[col]] = val
        
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    
    
    
def get_or_create_instance_Pi_Ci_one_period_docNONASH(setA_m_players, 
                                                      setB_m_players, 
                                                      setC_m_players, 
                                                      t_periods, 
                                                      scenario,
                                                      scenario_name,
                                                      path_to_arr_pl_M_T, 
                                                      used_instances):
    
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_docNONASH(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_docNONASH(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    

###############################################################################
#                   generate Pi, Ci One Period docNoNASH--> debut
###############################################################################

###############################################################################
#             generate Pi, Ci One Period doc2Players10000Etapes--> debut
###############################################################################
def generate_Pi_Ci_one_period_doc2Players10000Etapes(setA_m_players, 
                                        setB_m_players, 
                                        setC_m_players, 
                                        t_periods, 
                                        scenario):
    setA_id_players = [0]
    setC_id_players = [1]
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    num_pl_i = 0
    Si_t = 1
    Si_t_max = 3
    Ci_t = 3
    Pi_t = 1
    # update arrays cells with variables
    cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
            ("mode_i",""), ("state_i","")]
    for col, val in cols:
        arr_pl_M_T_vars[num_pl_i, t, 
                        AUTOMATE_INDEX_ATTRS[col]] = val
        
    num_pl_i = 1
    Si_t = 2
    Si_t_max = 3
    Ci_t = 1
    Pi_t = 3
    # update arrays cells with variables
    cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
            ("mode_i",""), ("state_i","")]
    for col, val in cols:
        arr_pl_M_T_vars[num_pl_i, t, 
                        AUTOMATE_INDEX_ATTRS[col]] = val
        
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    
    
    
def get_or_create_instance_Pi_Ci_one_period_doc2Players10000Etapes(setA_m_players, 
                                                      setB_m_players, 
                                                      setC_m_players, 
                                                      t_periods, 
                                                      scenario,
                                                      scenario_name,
                                                      path_to_arr_pl_M_T, 
                                                      used_instances):
    
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc2Players10000Etapes(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
            = generate_Pi_Ci_one_period_doc2Players10000Etapes(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    

###############################################################################
#             generate Pi, Ci One Period doc2Players10000Etapes--> FIN
###############################################################################

###############################################################################
#                   generate Pi, Ci One Period doc23--> debut
###############################################################################
def generate_Pi_Ci_one_period_doc23(setA_m_players=15, 
                                   setB_m_players=10, 
                                   setC_m_players=10, 
                                   t_periods=1, 
                                   scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA belonging to Deficit
             setB_m_players = number of players in setB belonging to Self
             setC_m_players = number of players in setC belonging to Surplus
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            prob = np.random.random(size=1)[0]
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_ABC[0]:                                             # setA = State1 or Deficit
                Si_t = 0
                Ci_t = 15
                Pi_t = np.random.randint(low=5, high=10, size=1)[0]
                
            elif setX == SET_ABC[1] and prob<0.5:                              # setB1 = State2 or Self
                Si_t = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=5, high=8, size=1)[0]
                
            elif setX == SET_ABC[1] and prob>=0.5:                              # setB1 State2 or Self
                Si_t = 8
                Ci_t = 31
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]
                
            elif setX == SET_ABC[2]:                                           # setC State3 or Surplus
                Si_t = 8
                Ci_t = 20
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    
  
def get_or_create_instance_Pi_Ci_one_period_doc23(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc23(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc23(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_one_period_doc23(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [5,10]; Cis = [15]; Sis = [0]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_ABC[1]:                                           # setB
                Pis_1 = [5,8]; Cis_1 = [10]; Sis_1 = [6]
                Pis_2 = [21,30]; Cis_2 = [31]; Sis_2 = [8]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if (Pi >= Pis_1[0] and Pi <= Pis_1[1]) \
                                    or (Pi >= Pis_2[0] and Pi <= Pis_2[1]) else 0
                cpt_t_Pi_nok += 1 if (Pi < Pis_1[0] or Pi > Pis_1[1]) \
                                     and (Pi < Pis_2[0] or Pi > Pis_2[1]) else 0
                cpt_t_Ci_ok += 1 if (Ci in Cis_1 or Ci in Cis_2)  else 0
                cpt_t_Ci_nok += 1 if (Ci not in Cis_1 and Ci not in Cis_2) else 0
                cpt_t_Si_ok += 1 if Si in Sis_1 or Si in Sis_2 else 0
                cpt_t_Si_nok += 1 if Si not in Sis_1 and Si not in Sis_2 else 0
                
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [21,30]; Cis = [20]; Sis = [8]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#                   generate Pi, Ci, Si One Period doc23 --> fin
###############################################################################

###############################################################################
#                   generate Pi, Ci One Period doc24--> debut
###############################################################################
def generate_Pi_Ci_one_period_doc24(setA_m_players=15, 
                                   setB_m_players=10, 
                                   setC_m_players=10, 
                                   t_periods=1, 
                                   scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA belonging to Deficit
             setB_m_players = number of players in setB belonging to Self
             setC_m_players = number of players in setC belonging to Surplus
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            prob = np.random.random(size=1)[0]
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_ABC[0]:                                             # setA = State1 or Deficit
                Si_t = 0
                Ci_t = 15
                Pi_t = np.random.randint(low=5, high=10, size=1)[0]
                
            elif setX == SET_ABC[1] and prob<0.5:                              # setB1 = State2 or Self
                Si_t = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=8, high=12, size=1)[0]
                
            elif setX == SET_ABC[1] and prob>=0.5:                              # setB1 State2 or Self
                Si_t = 8
                Ci_t = 31
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]
                
            elif setX == SET_ABC[2]:                                           # setC State3 or Surplus
                Si_t = 8
                Ci_t = 20
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    
  
def get_or_create_instance_Pi_Ci_one_period_doc24(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc24(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc24(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_one_period_doc24(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [5,10]; Cis = [15]; Sis = [0]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_ABC[1]:                                           # setB
                Pis_1 = [8,12]; Cis_1 = [10]; Sis_1 = [6]
                Pis_2 = [21,30]; Cis_2 = [31]; Sis_2 = [8]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if (Pi >= Pis_1[0] and Pi <= Pis_1[1]) \
                                    or (Pi >= Pis_2[0] and Pi <= Pis_2[1]) else 0
                cpt_t_Pi_nok += 1 if (Pi < Pis_1[0] or Pi > Pis_1[1]) \
                                     and (Pi < Pis_2[0] or Pi > Pis_2[1]) else 0
                cpt_t_Ci_ok += 1 if (Ci in Cis_1 or Ci in Cis_2)  else 0
                cpt_t_Ci_nok += 1 if (Ci not in Cis_1 and Ci not in Cis_2) else 0
                cpt_t_Si_ok += 1 if Si in Sis_1 or Si in Sis_2 else 0
                cpt_t_Si_nok += 1 if Si not in Sis_1 and Si not in Sis_2 else 0
                
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [21,30]; Cis = [20]; Sis = [8]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#                   generate Pi, Ci, Si One Period doc24 --> fin
###############################################################################

###############################################################################
#                   generate Pi, Ci One Period doc25--> debut
###############################################################################
def generate_Pi_Ci_one_period_doc25(setA_m_players=15, 
                                   setB_m_players=10, 
                                   setC_m_players=10, 
                                   t_periods=1, 
                                   scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA belonging to Deficit
             setB_m_players = number of players in setB belonging to Self
             setC_m_players = number of players in setC belonging to Surplus
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    Si_t_max = 50
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            prob = np.random.random(size=1)[0]
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_ABC[0]:                                             # setA = State1 or Deficit
                Si_t = 0
                Ci_t = 15
                Pi_t = np.random.randint(low=0, high=1, size=1)[0]
                
            elif setX == SET_ABC[1] and prob<0.5:                              # setB1 = State2 or Self
                Si_t = 0
                Ci_t = 10
                Pi_t = np.random.randint(low=8, high=12+1, size=1)[0]
                
            elif setX == SET_ABC[1] and prob>=0.5:                              # setB1 State2 or Self
                Si_t = 0
                Ci_t = 31
                Pi_t = np.random.randint(low=21, high=30+1, size=1)[0]
                
            elif setX == SET_ABC[2]:                                           # setC State3 or Surplus
                Si_t = 0
                Ci_t = 20
                Pi_t = np.random.randint(low=26, high=26+1, size=1)[0]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    
  
def get_or_create_instance_Pi_Ci_one_period_doc25(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc25(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_doc25(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_one_period_doc25(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [0,1]; Cis = [15]; Sis = [0]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_ABC[1]:                                           # setB
                Pis_1 = [8,12]; Cis_1 = [10]; Sis_1 = [0]
                Pis_2 = [21,30]; Cis_2 = [31]; Sis_2 = [0]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if (Pi >= Pis_1[0] and Pi <= Pis_1[1]) \
                                    or (Pi >= Pis_2[0] and Pi <= Pis_2[1]) else 0
                cpt_t_Pi_nok += 1 if (Pi < Pis_1[0] or Pi > Pis_1[1]) \
                                     and (Pi < Pis_2[0] or Pi > Pis_2[1]) else 0
                cpt_t_Ci_ok += 1 if (Ci in Cis_1 or Ci in Cis_2)  else 0
                cpt_t_Ci_nok += 1 if (Ci not in Cis_1 and Ci not in Cis_2) else 0
                cpt_t_Si_ok += 1 if Si in Sis_1 or Si in Sis_2 else 0
                cpt_t_Si_nok += 1 if Si not in Sis_1 and Si not in Sis_2 else 0
                
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [26,27]; Cis = [20]; Sis = [0]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#                   generate Pi, Ci, Si One Period doc24 --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc13 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc13(setA_m_players, 
                                           setB1_m_players, 
                                           setB2_m_players, 
                                           setC_m_players, 
                                           t_periods, 
                                           scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 13
                x = np.random.randint(low=2, high=6, size=1)[0]
                Pi_t = x + 2
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=25, high=35, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc13(setA_m_players, 
                                           setB1_m_players, 
                                           setB2_m_players, 
                                           setC_m_players, 
                                           t_periods, 
                                           scenario=None, 
                                           scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 14
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=10, high=12, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 22
                x = np.random.randint(low=18, high=22, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 26
                x = np.random.randint(low=8, high=18, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 20
                x = np.random.randint(low=24, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc13(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc13(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc13(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc13(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C_doc13(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C_doc13(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc13(arr_pl_M_T_vars_init, 
                                           scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [14]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10,12]; Cis = [10]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Pis = [18,22]; Cis = [22]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Pis = [8,18]; Cis = [26]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc13(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2+2,6+2]; Cis = [13]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [25,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc13--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc14 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc14(setA_m_players, 
                                           setB1_m_players, 
                                           setB2_m_players, 
                                           setC_m_players, 
                                           t_periods, 
                                           scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 13
                x = np.random.randint(low=2, high=6, size=1)[0]
                Pi_t = x + 2
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=25, high=35, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc14(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 14
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=10, high=12, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 22
                x = np.random.randint(low=18, high=22, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 26
                x = np.random.randint(low=8, high=18, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc14(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc14(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc14(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc14(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C_doc14(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C_doc14(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc14(arr_pl_M_T_vars_init, 
                                           scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [14]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10,12]; Cis = [10]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Pis = [18,22]; Cis = [22]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Pis = [8,18]; Cis = [26]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc14(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2+2,6+2]; Cis = [13]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [25,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc14--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc15 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc15(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc15(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 22
                x = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc15(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc15(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc15(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc15(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc15(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc15(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc15(arr_pl_M_T_vars_init, 
                                           scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,24]; Cis = [22]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc15(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10]; Cis = [8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [18,24]; Cis = [22]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc15--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc16 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc16(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc16(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 22
                x = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc16(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc16(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc16(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc16(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc16(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc16(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc16(arr_pl_M_T_vars_init, 
                                           scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,24]; Cis = [22]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc16(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10]; Cis = [8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [18,24]; Cis = [22]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc16--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc17 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc17(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 20
                x = np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc17(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Ci_t = 22
                x = np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                Ci_t = 20
                x = 30 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc17(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc17(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc17(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc17(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc17(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc17(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc17(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,24]; Cis = [22]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,30]; Cis = [20]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc17(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10]; Cis = [8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [18,24]; Cis = [22]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc17--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc18 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc18(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = 3 #np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = 18 #np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 20
                x = 25 #np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc18(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 10
                Pi_t = 3
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Ci_t = 22
                Pi_t = 18
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                Ci_t = 20
                x = 25 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc18(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc18(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc18(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc18(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc18(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc18(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc18(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [10]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,18]; Cis = [22]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [25,25]; Cis = [20]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc18(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10]; Cis = [8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [18,18]; Cis = [22]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [25,25]; Cis = [20]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc18--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc19 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc19(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 12
                x = 3 #np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = 18 #np.random.randint(low=18, high=24, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 20
                x = 25 #np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc19(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 12
                Pi_t = 3
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Ci_t = 22
                Pi_t = 18
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                Ci_t = 20
                x = 25 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc19(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc19(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc19(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc19(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc19(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc19(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc19(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10,10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,18]; Cis = [22]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [25,25]; Cis = [20]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc19(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10, 10]; Cis = [8,8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [18,18]; Cis = [22,22]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [25,25]; Cis = [20]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc19--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc20 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc20(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 12
                x = 3 #np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 8
                Pi_t = 10
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 18
                y = np.random.randint(low=15, high=18, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 18
                x = 20 #np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc20(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 12
                Pi_t = 3
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 8
                Pi_t = 10
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Ci_t = 18
                Pi_t = np.random.randint(low=15, high=18, size=1)[0]
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                Ci_t = 18
                x = 20 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc20(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc20(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc20(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10,10]; Cis = [8]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [15,18]; Cis = [18]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [20,20]; Cis = [18]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc20(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10, 10]; Cis = [8,8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [15,18]; Cis = [18,18]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [20,20]; Cis = [18]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc20--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc22 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc22(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 10
                Pi_t = np.random.randint(low=8, high=10, size=1)[0]
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = np.random.randint(low=18, high=22, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 22
                x = 26 #np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc22(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=2, high=4, size=1)[0]
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=8, high=10, size=1)[0]
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Ci_t = 22
                Pi_t = np.random.randint(low=18, high=22, size=1)[0]
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                Ci_t = 20
                x = 26 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc22(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc22(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc22(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [10]; Sis = [3]; Sis_max=[6]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [8,10]; Cis = [10]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,22]; Cis = [22]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [26,26]; Cis = [20]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc22(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10, 10]; Cis = [8,8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [15,18]; Cis = [18,18]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [20,20]; Cis = [18]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc22--> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C doc23 --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C_doc23(setA_m_players, 
                                                setB1_m_players, 
                                                setB2_m_players, 
                                                setC_m_players, 
                                                t_periods, 
                                                scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Ci_t = 10
                Pi_t = np.random.randint(low=8, high=10, size=1)[0]
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Ci_t = 22
                y = np.random.randint(low=18, high=22, size=1)[0]
                Pi_t = y
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 22
                x = 26 #np.random.randint(low=30, high=40, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C_doc23(setA_m_players, 
                                               setB1_m_players, 
                                               setB2_m_players, 
                                               setC_m_players, 
                                               t_periods, 
                                               scenario=None, 
                                               scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 6
                Pi_t = np.random.randint(low=2, high=4, size=1)[0]
                Ci_t = 10 if scenario_name == "scenario2" else 20
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 6
                Pi_t = np.random.randint(low=8, high=12, size=1)[0]
                Ci_t = 10 if scenario_name == "scenario2" else 12
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 10
                Si_t_max = 15
                Pi_t = np.random.randint(low=18, high=22, size=1)[0]
                Ci_t = 22 if scenario_name == "scenario2" else 22
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 15
                x = 26 #np.random.randint(low=30, high=30, size=1)[0]
                Pi_t = x
                Ci_t = 20 if scenario_name == "scenario2" else 20
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc23(setA_m_players, 
                                                               setB1_m_players, 
                                                               setB2_m_players, 
                                                               setC_m_players, 
                                                               t_periods, 
                                                               scenario, 
                                                               scenario_name,
                                                               path_to_arr_pl_M_T, 
                                                               used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc23(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc23(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods,
                    scenario=scenario, 
                    scenario_name=scenario_name)

        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C_doc23(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C_doc20(
                    setA_m_players, 
                    setB1_m_players, 
                    setB2_m_players, 
                    setC_m_players, 
                    t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C_doc23(arr_pl_M_T_vars_init, 
                                                 scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Sis = [3]; Sis_max=[6]
                Cis = [10] if scenario_name == "scenario2" else [20]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [8,12]; Cis = [10]; Sis = [4]; Sis_max=[6]
                Cis = [10] if scenario_name == "scenario2" else [12]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Pis = [18,22]; Cis = [22]; Sis = [10]; Sis_max=[15]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [26,26]; Cis = [20]; Sis = [10]; Sis_max=[15]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C_doc23(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [3,3]; Cis = [12]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [10, 10]; Cis = [8,8]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [15,18]; Cis = [18,18]; Sis = [10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [20,20]; Cis = [18]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C doc23 --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AC --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAC(setA_m_players, 
                                    setC_m_players, 
                                    t_periods, 
                                    scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                    )
            ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Ci_t = 20
                Pi_t = 10
                state_i = STATES[0]
                
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Ci_t = 24
                Pi_t = 32
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAC(setA_m_players,  
                                    setC_m_players, 
                                    t_periods, 
                                    scenario=None,
                                    scenario_name="scenario0"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                       )
                ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 20
                Pi_t = 10
                # player' set at t+1 prob_A_A = 0.6; prob_A_C = 0.4 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_A_A,
                                                             prob_A_C])
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 25
                Pi_t = 35
                # player' set at t+1 prob_C_A = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_C_A,
                                                             prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(setA_m_players,
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC(setA_m_players,  
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAC(setA_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAC(arr_pl_M_T_vars_init, scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [10]; Cis = [20]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [35]; Cis = [25]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAC(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [10]; Cis = [20]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [32]; Cis = [24]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AC doc23--> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAC_doc23(setA_m_players, 
                                          setC_m_players, 
                                          t_periods, 
                                          scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                    )
            ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Ci_t = 20
                Pi_t = 10
                state_i = STATES[0]
                
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Ci_t = 24
                Pi_t = 32
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAC_doc23(setA_m_players,  
                                    setC_m_players, 
                                    t_periods, 
                                    scenario=None,
                                    scenario_name="scenario0"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                       )
                ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 10
                Pi_t = 0
                # player' set at t+1 prob_A_A = 0.6; prob_A_C = 0.4 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_A_A,
                                                             prob_A_C])
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 10
                Pi_t = 20
                # player' set at t+1 prob_C_A = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_C_A,
                                                             prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC_doc23(setA_m_players,
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC_doc23(setA_m_players,  
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC_doc23(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAC_doc23(setA_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {Deficit, Self} : set of players' states 
    set2 = {Self, Surplus}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods, scenario_name)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAC_doc23(arr_pl_M_T_vars_init, scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [0]; Cis = [10]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [20]; Cis = [10]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAC_doc23(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [10]; Cis = [20]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [32]; Cis = [24]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C --> fin
###############################################################################

##############################################################################
#           look for whether pli is balanced or not --> debut  
##############################################################################
def balanced_player(pl_i, thres=0.1, dbg=False):
    """
    verify if pl_i is whether balanced or unbalanced

    Parameters
    ----------
    pl_i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Pi = pl_i.get_Pi(); Ci = pl_i.get_Ci(); Si_new = pl_i.get_Si(); 
    Si_max = pl_i.get_Si_max(); R_i_old = pl_i.get_R_i_old()
    state_i = pl_i.get_state_i(); 
    mode_i = pl_i.get_mode_i()
    cons_i = pl_i.get_cons_i(); prod_i = pl_i.get_prod_i()
    Si_old = pl_i.get_Si_old()
    
    if dbg:
        print("_____ balanced_player Pi={}, Ci={}, Si={}, Si_max={}, state_i={}, mode_i={}"\
              .format(pl_i.get_Pi(), pl_i.get_Ci(), pl_i.get_Si(), 
                      pl_i.get_Si_max(), pl_i.get_state_i(), 
                      pl_i.get_mode_i())) 
    boolean = None
    if state_i == "Deficit" and mode_i == "CONS+":
        boolean = True if np.abs(Pi+(Si_old-Si_new)+cons_i - Ci)<thres else False
        formule = "Pi+(Si_old-Si_new)+cons_i - Ci"
        res = Pi+(Si_old-Si_new)+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Ci':np.round(Ci,2),
                'Si_new':np.round(Si_new,2), 'Si_max':np.round(Si_max,2), 
                'cons_i':np.round(cons_i,2), 'R_i_old': np.round(R_i_old,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "Deficit" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "Self" and mode_i == "DIS":
        boolean = True if np.abs(Pi+(Si_old-Si_new) - Ci)<thres else False
        formule = "Pi+(Si_old-Si_new) - Ci"
        res = Pi+(Si_old-Si_new) - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "Self" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "Surplus" and mode_i == "PROD":
        boolean = True if np.abs(Pi - Ci-prod_i)<thres else False
        formule = "Pi - Ci-prod_i"
        res = Pi - Ci-prod_i
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                "res": res}
    elif state_i == "Surplus" and mode_i == "DIS":
        boolean = True if np.abs(Pi - Ci-(Si_max-Si_old)-prod_i)<thres else False
        formule = "Pi - Ci-(Si_max-Si_old)-prod_i"
        res = Pi - Ci-(Si_max-Si_old)-prod_i
        dico = {'Pi': np.round(Pi,2), 'Si_new': np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                    "res": res, }
    return boolean, formule
##############################################################################
#           look for whether pli is balanced or not --> fin 
##############################################################################
