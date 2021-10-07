# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:12:43 2021

@author: jwehounou
"""

import time
import math
import numpy as np

import fonctions_auxiliaires as fct_aux

N_INSTANCE = 10

#MODES = ("CONS","PROD","DIS")

# STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
# STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
# STATE3_STRATS = ("DIS", "PROD") 

class Player:
    
    cpt_player =  0
    
    def __init__(self, Pi, Ci, Si, Si_max, 
                 gamma_i, prod_i, cons_i, r_i, state_i):
        self.name = ("").join(["a",str(self.cpt_player)])
        self.Pi = Pi
        self.Ci = Ci
        self.Si = Si
        self.Si_max = Si_max
        self.gamma_i = gamma_i
        Player.cpt_player += 1
        
        # variables depend on a decision of the instance
        self.prod_i = prod_i
        self.cons_i = cons_i
        self.R_i_old = 0
        self.Si_old = 0
        self.Si_minus = 0
        self.Si_plus = 0
        self.r_i = r_i
        self.state_i = state_i
        self.mode_i = ""
        
        
    #--------------------------------------------------------------------------
    #           definition of caracteristics of an agent
    #--------------------------------------------------------------------------
    def get_Pi(self):
        """
        return the value of quantity of production
        """
        return self.Pi
    
    def set_Pi(self, new_Pi, update=False):
        """
        return the new quantity of production or the energy quantity 
        to add from the last quantity of production.
        
        self.Pi = new_Pi if update==True else self.Pi + new_Pi
        """
        self.Pi = (update==False)*new_Pi + (update==True)*(self.Pi + new_Pi)
            
    def get_Ci(self):
        """
        return the quantity of consumption 
        """
        return self.Ci
    
    def set_Ci(self, new_Ci, update=False):
        """
        return the new quantity of consumption or the energy quantity 
        to add from the last quantity of production.
        
        self.Ci = new_Ci if update==True else self.Ci + new_Ci
        """
        self.Ci = (update==False)*new_Ci + (update==True)*(self.Ci + new_Ci)
        
    def get_Si(self):
        """
        return the value of quantity of battery storage
        """
        return self.Si
    
    def set_Si(self, new_Si, update=False):
        """
        return the new quantity of battery storage or the energy quantity 
        to add from the last quantity of storage.
        
        self.Si = new_Si if update==True else self.Si + new_Si
        """
        self.Si = (update==False)*new_Si + (update==True)*(self.Si + new_Si)
        
    def get_Si_max(self):
        """
        return the value of quantity of production
        """
        return self.Si_max
    
    def set_Si_max(self, new_Si_max, update=False):
        """
        return the new quantity of the maximum battery storage or the energy 
        quantity to add from the last quantity of teh maximum storage.
        
        self.Si = new_Pi if update==True else self.Pi + new_Pi
        """
        self.Si_max = (update==False)*new_Si_max \
                        + (update==True)*(self.Si_max + new_Si_max)
                        
    def get_R_i_old(self):
        """
        return the reserv amount before updating Si

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.R_i_old
    
    def set_R_i_old(self, new_R_i_old, update=False):
        """
        turn the old reserv amount into new_R_i_old if update=False else add 
        new_R_i_old  to the last value

        Parameters
        ----------
        new_R_i_old : float
            DESCRIPTION.
        update : booelan, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.R_i_old = (update==False)*new_R_i_old \
                        + (update==True)*(self.R_i_old + new_R_i_old)
                        
    def get_Si_old(self):
        """
        return the battery storage amount before updating Si

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.Si_old
    
    def set_Si_old(self, new_Si_old, update=False):
        """
        turn the old the battery storage amount into new_Si_old if update=False else add 
        new_Si_old to the last value

        Parameters
        ----------
        new_Si_old : float
            DESCRIPTION.
        update : booelan, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.Si_old = (update==False)*new_Si_old \
                        + (update==True)*(self.Si_old + new_Si_old)
                        
    def get_Si_minus(self):
        """
        return the min battery storage amount between 2 modes of one state state_i

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.Si_minus
    
    def set_Si_minus(self, new_Si_minus, update=False):
        """
        turn the old min battery storage amount into new_Si_minus if update=False else add 
        new_Si_minus to the last value

        Parameters
        ----------
        new_Si_minus : float
            DESCRIPTION.
        update : boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.Si_minus = (update==False)*new_Si_minus \
                        + (update==True)*(self.Si_minus + new_Si_minus)
                        
    def get_Si_plus(self):
        """
        return the max battery storage amount between 2 modes of one state state_i

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.Si_plus
    
    def set_Si_plus(self, new_Si_plus, update=False):
        """
        turn the old max battery storage amount into new_Si_plus if update=False else add 
        new_Si_plus to the last value

        Parameters
        ----------
        new_Si_plus : float
            DESCRIPTION.
        update : boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.Si_plus = (update==False)*new_Si_plus \
                        + (update==True)*(self.Si_plus + new_Si_plus)
                        
    def get_gamma_i(self):
        """
        gamma denotes the behaviour of the agent to store energy or not. 
        the value implies the price of purchase/sell energy.
        return the value of the behaviour  
    
        NB: if gamma_i = np.inf, the agent has a random behaviour ie he has 
            50% of chance to store energy
        """
        return self.gamma_i
    
    def set_gamma_i(self, new_gamma_i):
        """
        return the new value of the behaviour or the energy 
        quantity to add from the last quantity of the maximum storage.
        
        NB: if gamma_i = np.inf, the agent has a random behaviour ie he has 
            50% of chance to store energy
        """
        self.gamma_i = new_gamma_i 
        
    def get_prod_i(self):
        """
        return the production amount to export to SG 

        Returns
        -------
        an integer or float.

        """
        return self.prod_i
    
    def set_prod_i(self, new_prod_i, update=False):
        """
        turn the production amount into new_prod_i if update=False else add 
        new_prod_i  to the last value

        Parameters
        ----------
        new_prod_i : float
            DESCRIPTION.
        update : booelan, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.prod_i = (update==False)*new_prod_i \
                        + (update==True)*(self.prod_i + new_prod_i)
                        
    def get_cons_i(self):
        """
        return the consumption amount to import from HP to SG 

        Returns
        -------
        an integer or float.

        """
        return self.cons_i
    
    def set_cons_i(self, new_cons_i, update=False):
        """
        turn the consumption amount into new_cons_i if update=False else add 
        new_cons_i  to the last value

        Parameters
        ----------
        new_cons_i : float
            DESCRIPTION.
        update : booelan, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.cons_i = (update==False)*new_cons_i \
                        + (update==True)*(self.cons_i + new_cons_i)
                        
    def get_r_i(self):
        """
        return the consumption amount to import from HP to SG 

        Returns
        -------
        an integer or float.

        """
        return self.r_i
    
    def set_r_i(self, new_r_i, update=False):
        """
        turn the amount of energy stored (or preserved by a player) into 
        new_r_i if update=False else add new_ri_i  to the last value

        Parameters
        ----------
        new_r_i : float
            DESCRIPTION.
        update : booelan, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        float.

        """
        self.r_i = (update==False)*new_r_i \
                        + (update==True)*(self.r_i + new_r_i)
                        
    def get_state_i(self):
        """
        return the state of the player

        Returns
        -------
        None.

        """
        return self.state_i
    
    def find_out_state_i(self):
        """
        return the state of the player depending on the operation conditions 

        Returns
        -------
        a string.

        """
        if self.Pi + self.Si <= self.Ci:
            self.state_i = fct_aux.STATES[0]
        elif self.Pi + self.Si > self.Ci and self.Pi <= self.Ci:
            self.state_i = fct_aux.STATES[1]
        elif self.Pi >= self.Ci:
            self.state_i = fct_aux.STATES[2]
        else:
            self.state_i = None
        return self.state_i
    
    def set_state_i(self, new_state_i):
        """
        turn the player state into 
        new_state_i

        Parameters
        ----------
        new_state_i : string
            DESCRIPTION.

        Returns
        -------
        a string.

        """
        self.state_i = new_state_i 
        
    def get_mode_i(self):
        """
        return the mode of player i

        Returns
        -------
        a string.

        """
        return self.mode_i
    
    def set_mode_i(self, new_mode_i):
        """
        update the mode of player i

        Returns
        -------
        None.

        """
        self.mode_i = new_mode_i
    #--------------------------------------------------------------------------
    #           definition of functions of an agent
    #--------------------------------------------------------------------------
    def select_mode_i(self, p_i=0.5, thres=None):
        """
        select randomly a mode of an agent i
        
        Parameters
        ----------
        p_i: float [0,1], 
            DESCRIPTION. The default is 0.5
            probability to choose the first item in state mode

        Returns
        -------
        update variable mode_i containing
        string value if state_i != None or None if state_i == None

        """
        mode_i = None
        rd_num =  np.random.choice([0,1], p=[p_i, 1-p_i])
        if self.state_i == None:
            mode_i = None
        elif self.state_i == fct_aux.STATES[0]:
            mode_i = fct_aux.STATE1_STRATS[rd_num]
        elif self.state_i == fct_aux.STATES[1]:
            mode_i = fct_aux.STATE2_STRATS[rd_num]
        elif self.state_i == fct_aux.STATES[2]:
            mode_i = fct_aux.STATE3_STRATS[rd_num]
        self.mode_i = mode_i
        
    def update_prod_cons_r_i(self):
        """
        compute prod_i, cons_i and r_i following the characteristics of agent i 

        Returns
        -------
        update variable prod_i, cons_i, r_i containing
        float value if state != None or np.nan if state == None.

        """
        
        # compute preserved stock r_i
        if self.mode_i == "CONS+":
            self.r_i = 0
        elif self.mode_i == "CONS-":
            self.r_i = self.Si
        elif self.mode_i == "PROD":
            self.r_i = 0 #self.Si
        elif self.mode_i == "DIS" and self.state_i ==  fct_aux.STATES[1]:
            self.r_i = self.Si - (self.Ci - self.Pi)
        elif self.mode_i == "DIS" and self.state_i ==  fct_aux.STATES[2]:
            self.r_i = min(self.Si_max - self.Si, self.Pi - self.Ci)
        
        if self.state_i ==  fct_aux.STATES[0]:                                 # Deficit
            self.prod_i = 0
            self.cons_i = (self.mode_i == "CONS+")*(self.Ci - (self.Pi + self.Si)) \
                            + (self.mode_i == "CONS-")*(self.Ci - self.Pi)
            self.Si_old = (self.mode_i == "CONS+")*self.Si \
                            + (self.mode_i == "CONS-")*self.Si_old
            self.Si = (self.mode_i == "CONS+")*0 \
                        + (self.mode_i == "CONS-")*self.Si
            R_i = self.Si_max - self.Si
            
            
        elif self.state_i ==  fct_aux.STATES[1]:                               # Self
            self.prod_i = 0
            self.cons_i = (self.mode_i == "DIS")*0 \
                            + (self.mode_i == "CONS-")*(self.Ci - self.Pi)
            self.Si_old = (self.mode_i == "DIS")*self.Si \
                            + (self.mode_i == "CONS-")*self.Si_old
            self.Si = (self.mode_i == "DIS")*(
                        max(0,self.Si - (self.Ci - self.Pi))) \
                        + (self.mode_i == "CONS-")*self.Si
    
        elif self.state_i ==  fct_aux.STATES[2]:                               # Surplus
            self.cons_i = 0
            if self.Pi == self.Ci:
                self.Si = self.Si
                self.Si_old = self.Si_old
                self.prod_i = 0
            else:
                R_i = self.Si_max - self.Si
                self.Si_old = (self.mode_i == "DIS")*self.Si \
                                + (self.mode_i == "PROD")*self.Si_old
                self.Si = (self.mode_i == "DIS") \
                                *(min(self.Si_max, self.Si + (self.Pi - self.Ci))) \
                            + (self.mode_i == "PROD")*self.Si
                self.prod_i = (self.mode_i == "PROD")*(self.Pi - self.Ci)\
                               + (self.mode_i == "DIS") \
                                   *fct_aux.fct_positive(sum([self.Pi]), 
                                                         sum([self.Ci, R_i]))
            
        else:
            # state_i = mode_i = None
            self.prod_i = np.nan
            self.cons_i = np.nan
            self.r_i = np.nan
        
        
        
    def select_storage_politic(self, Ci_t_plus_1, Pi_t_plus_1, 
                               pi_0_plus, pi_0_minus, 
                               pi_hp_plus, pi_hp_minus):
        """
        choose the storage politic gamma_i of the player i 
        following the rules on the smartgrid system model document 
        (see storage politic).

        Parameters
        ----------
        Ci_t_plus_1 : float
            DESCRIPTION.
        Pi_t_plus_1 : float
            DESCRIPTION.
        pi_0_plus : float
            DESCRIPTION.
        pi_0_minus : float
            DESCRIPTION.
        pi_hp_plus : float
            DESCRIPTION.
        pi_hp_minus : float
            DESCRIPTION.

        Returns
        -------
        a float if conditions are not respected or np.inf else.

        """
        Si_minus, Si_plus = 0, 0
        X, Y = 0, 0
        if self.state_i == fct_aux.STATES[0]:
            # Si_minus = 0 if self.mode_i == "CONS+" else 0
            # Si_plus = self.get_Si() if self.mode_i == "CONS-" else 0
            if self.mode_i == "CONS+":
                Si_minus = 0;
                Si_plus = self.Si
            else:
                # CONS-
                Si_minus = 0;
                Si_plus = self.Si
            X = pi_0_minus
            Y = pi_hp_minus
        elif self.state_i == fct_aux.STATES[1]:
            # Si_minus = self.get_Si() - (self.get_Ci() - self.get_Pi()) \
            #     if self.mode_i == "DIS" else 0
            # Si_plus = self.get_Si() if self.mode_i == "CONS-" else 0
            if self.mode_i == "DIS":
                Si_minus = self.Si - (self.Ci - self.Pi)
                Si_plus = self.Si
            else:
                # CONS-
                Si_minus = 0
                Si_plus = self.Si
            X = pi_0_minus
            Y = pi_hp_minus
        elif self.state_i == fct_aux.STATES[2]:
            # Si_minus = self.get_Si() if self.mode_i == "PROD" else 0
            # Si_plus = max(self.get_Si_max(), 
            #               self.get_Si() + (self.get_Pi() - self.get_Ci()))
            if self.mode_i == "PROD":
                Si_minus = self.Si
                Si_plus = self.Si_max
            else:
                # DIS
                Si_minus = 0
                Si_plus = max(self.Si_max, 
                              self.Si 
                              + (self.Pi 
                                 - self.Ci
                                 )
                              )
            X = pi_0_plus
            Y = pi_hp_plus
        else:
            Si_minus, Si_plus = np.inf, np.inf
            X, Y = np.inf, np.inf

        self.Si_minus = Si_minus
        self.Si_plus = Si_plus


        #print("gamma_i Si_minus={} < Si_plus={} {}".format(round(Si_minus,2), round(Si_plus,2), Si_minus <= Si_plus))
        if Si_minus > Si_plus:
            print("Si_minus={} Si_plus={}".format(Si_minus, Si_plus))
            print("Si_old={}, Si={}, Si_max={} state={}, mode={}".format(self.Si_old, 
                    self.Si, self.Si_max, self.state_i, self.mode_i))
        if fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_minus:
            self.set_gamma_i(X-1)
        elif fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_plus:
            self.set_gamma_i(Y+1)
        elif fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_minus and \
            fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_plus:
                res = (fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1)- Si_minus)\
                       / (Si_plus - Si_minus)
                Z = X + (Y-X)*res
                
                #print("res={}, Z={}, X={}, Y={}".format(res, Z, X, Y))
                # if (X>Z or Z>Y):
                #     print("X>Z>Y: {}".format( X>Z or Z>Y ))
                self.set_gamma_i(math.floor(Z))   
        else:
            self.set_gamma_i(np.inf) 
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------
def test_class_player():
    Cis = np.random.uniform(low=1, high=30, size=(1,N_INSTANCE))
    
    low = 0; high = 0.3
    Cis, Pis, Si_maxs, Sis = fct_aux.generate_Cis_Pis_Sis(
                                    n_items = N_INSTANCE, 
                                    low_1 = 1, 
                                    high_1 = 30,
                                    low_2 = low,
                                    high_2 = high
                                    )
    
    bool_pis = (Pis/Cis> low).all() and (Pis/Cis<= high).all()
    bool_si_maxs = (Si_maxs/Pis> low).all() and (Si_maxs/Pis<= high).all()
    bool_sis = (Sis/Si_maxs> low).all() and (Sis/Si_maxs<= high).all()
    print("bool_pis={}, bool_si_maxs={}, bool_sis={}".format(
        bool_pis, bool_si_maxs, bool_sis))
    
    gamma_is = np.zeros(shape=(1, N_INSTANCE))
    prod_is = np.zeros(shape=(1, N_INSTANCE))
    cons_is = np.zeros(shape=(1, N_INSTANCE))
    r_is = np.zeros(shape=(1, N_INSTANCE))
    state_is = np.array([None]*N_INSTANCE).reshape((1,-1))
    print("Pis={}, Cis={}, Sis={}, Si_maxs={}, prod_is={}, cons_is={}, gamma_is={}, state_is={}, r_is={}".format(
            Pis.shape, Cis.shape, Sis.shape, Si_maxs.shape, prod_is.shape, cons_is.shape, 
            gamma_is.shape, r_is.shape, state_is.shape))
    
    nb_test = 11; nb_ok = 0;
    for ag in np.concatenate((Pis, Cis, Sis, Si_maxs, gamma_is, prod_is, 
                              cons_is, r_is, state_is)).T:
        pl = Player(*ag)
        
        nb = np.random.randint(1,30); OK = 0
        #Ci
        oldCi = pl.get_Ci(); pl.set_Ci(nb, True)
        OK = OK+1 if pl.get_Ci() == oldCi + nb else OK-1
        pl.set_Ci(oldCi, False)
        #Pi
        oldPi = pl.get_Pi(); pl.set_Pi(nb, True)
        OK = OK+1 if pl.get_Pi() == oldPi+nb else OK-1
        pl.set_Pi(oldPi, False)
        #Si_max
        oldSi_max = pl.get_Si_max(); pl.set_Si_max(nb, True)
        OK = OK+1 if pl.get_Si_max() == oldSi_max + nb else OK-1
        pl.set_Si_max(oldPi, False)
        #Si
        oldSi = pl.get_Si(); pl.set_Si(nb, True)
        OK = OK+1 if pl.get_Si() == oldSi + nb else OK-1
        pl.set_Si(oldSi, False)
        #gamma_i
        oldGamma_i = pl.get_gamma_i(); pl.set_gamma_i(nb)
        OK = OK+1 if pl.get_gamma_i() == nb else OK-1
        pl.set_gamma_i(oldGamma_i)
        #prod_i
        oldProd_i = pl.get_prod_i(); pl.set_prod_i(nb, True)
        OK = OK+1 if pl.get_prod_i() == oldProd_i + nb else OK-1
        pl.set_prod_i(oldProd_i, False)
        #cons_i
        oldCons_i = pl.get_cons_i(); pl.set_cons_i(nb, True)
        OK = OK+1 if pl.get_cons_i() == oldCons_i + nb else OK-1
        pl.set_cons_i(oldCons_i, False)
        #r_i
        oldr_i = pl.get_r_i(); pl.set_r_i(nb, True)
        OK = OK+1 if pl.get_r_i() == oldr_i + nb else OK-1
        pl.set_r_i(oldr_i, False)
        #state_i
        OK_state = 0; OK_state_none = 0
        state_i = pl.find_out_state_i()
        if state_i == fct_aux.STATES[0] and pl.get_Pi() + pl.get_Si() <= pl.get_Ci():
            OK_state += 1
        elif state_i == fct_aux.STATES[1] and pl.get_Pi() + pl.get_Si() > pl.get_Ci() \
            and pl.get_Pi() < pl.get_Ci():
                OK_state += 1
        elif state_i == fct_aux.STATES[2] and pl.get_Pi() > pl.get_Ci():
            OK_state += 1
        elif state_i == None:
            OK_state_none += 1
            OK_state += -1
        # mode i
        if pl.state_i in fct_aux.STATES and \
            pl.mode_i is not None and \
            pl.prod_i is not np.nan  and \
            pl.cons_i is not np.nan and \
            pl.r_i is not np.nan:
                OK += 1
        # behaviour of gamma_i
        pl.select_storage_politic(Ci_t_plus_1=0, Pi_t_plus_1=0, 
                                  pi_0_plus=np.random.uniform(1,10), 
                                  pi_0_minus=np.random.uniform(1,10), 
                                  pi_hp_plus=np.random.uniform(1,10), 
                                  pi_hp_minus=np.random.uniform(1,10))
        if pl.get_gamma_i() is not np.inf:
            OK += 1
            print("pl {}: gamma_i={}".format(pl.name, pl.get_gamma_i()))
        
        # afficher indicateurs et calcul nb_ok
        nb_ok += (OK + OK_state)/nb_test
        
    # les afficher ces indicateurs rp
    rp = round(nb_ok/N_INSTANCE, 3)
    print("rapport execution rp={}".format(rp))
    

def test_class_player_r_i():
    
    import os
    m_players=4; t_periods=2; 

    
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"]); 
    used_instances=True
    
    arr_pl_M_T_vars = fct_aux.get_or_create_instance_2_4players(m_players, t_periods,
                                      path_to_arr_pl_M_T, 
                                      used_instances)
    
    t = 1
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars[num_pl_i, t,
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars[num_pl_i, t,
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']] 
        Si_max = arr_pl_M_T_vars[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        state_i = arr_pl_M_T_vars[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # select mode for player num_pl_i
        p_i_t_k = 0.5
        pl_i.select_mode_i(p_i=p_i_t_k)
        
        # compute cons, prod, r_i
        pl_i.update_prod_cons_r_i()
        
        print("player pl_{}: {}, mode_i={}, gamma={}, Si_old={}, Si={}, r_i={}, prod_i={}, cons_i={}, Pi={}, Ci={}, Si_max={}".format(
            num_pl_i, pl_i.get_state_i(), pl_i.get_mode_i(), pl_i.get_gamma_i(), pl_i.get_Si_old(), 
            pl_i.get_Si(), pl_i.get_r_i(), pl_i.get_prod_i(),  pl_i.get_cons_i(),  Pi, Ci, Si_max))

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    # test_class_player()
    # y = test_merge_vars()
    test_class_player_r_i()
    print("classe player runtime = {}".format(time.time() - ti))


