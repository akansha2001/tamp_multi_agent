
# TAMPURA
import random
from dataclasses import dataclass, field
from typing import List,Dict

import copy
import itertools 
import time
from tampura.policies.policy import save_config, RolloutHistory, save_run_data

import numpy as np
import os
from tampura.environment import TampuraEnv
from tampura.spec import ProblemSpec
from tampura.structs import (
    AbstractBelief,
    ActionSchema,
    StreamSchema,
    AliasStore,
    Belief,
    NoOp,
    Predicate,
    State,
    effect_from_execute_fn,
    Observation,
    AbstractBeliefSet,
)
import logging 
from tampura.symbolic import OBJ, Atom, ForAll, Not, Exists, Or, And, OneOf, eval_expr
from tampura.policies.tampura_policy import TampuraPolicy
from tampura.config.config import load_config, setup_logger

from pick_successes import SIMPLE_PICK_EGO_SIM, CABINET_PICK_EGO_SIM

ROB = "robot_"
REG = "region_"
MUG = "mug"
DOOR = "door"
REGIONS = [f"{REG}{MUG}",f"{REG}{DOOR}",f"{REG}stable_mug"]
ACTION_NAMES = ["transit_action","transfer_action","pick_action","place_action","open_action","close_action","nothing_action"]

# problem specification: try with just one robot to demonstrate how overall cost increases
ROBOTS=[f"{ROB}1",f"{ROB}2"]
ROB_REGIONS = {ROBOTS[0]:REGIONS[-1],ROBOTS[1]:REGIONS[-1]} # long horizon: combinatorial explosion
# ROB_REGIONS = {ROBOTS[0]:REGIONS[1],ROBOTS[1]:REGIONS[0]} # short horizon: kind of works?
OBJ_REGIONS={MUG:REGIONS[0]}
# probability of success for open/close by ego
OPEN_EGO = 0.2
CLOSE_EGO = 0.3
# higher num_samples needed to learn true transition model

# Test 
GOAL = And([Exists(Atom("holding",["?rob",MUG]),["?rob"],["robot"]),Not(Atom("open",[DOOR]))])


# State of the environment
@dataclass
class EnvState(State):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    open_door: bool = field(default_factory=lambda: False)
    rob_regions: Dict[str,str] = field(default_factory=lambda:{})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    next_actions: List[str] = field(default_factory=lambda: [])
    
# Observation space
@dataclass
class EnvObservation(Observation):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    open_door: bool = field(default_factory=lambda: False)
    rob_regions: Dict[str,str] = field(default_factory=lambda:{})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    next_actions: List[str] = field(default_factory=lambda: [])

# Belief space
class EnvBelief(Belief):
    def __init__(self, holding={},open_door=False,rob_regions={},obj_regions={},next_actions=[]):
        # true state
        self.holding = holding
        self.open_door = open_door
        self.rob_regions = rob_regions
        self.obj_regions = obj_regions
        self.next_actions = next_actions
        

    def update(self, a, o, s):
        
        # dictionary mutations are IN-PLACE!!! use .copy()!!
        holding = self.holding.copy() 
        open_door = self.open_door
        rob_regions = self.rob_regions.copy()
        obj_regions = self.obj_regions.copy()
        next_actions = self.next_actions
        
        
        # get argument index for ego agent
        
        a_other_name,a_ego_name = a.name.split("*")
        
        if a_other_name == "transfer_other":
            nargs_other = 4
        elif a_other_name == "nothing_other" or a_other_name == "open_other" or a_other_name == "close_other":
            nargs_other = 1
        else:
            nargs_other = 3
            
        a_ego_args = a.args[nargs_other:]
        
        
        # special cases
        # case 1: place, pick
        if a_other_name == "place_other" and a_ego_name == "pick_ego":
            
            holding = o.holding
            obj_regions = o.obj_regions
            
        # case 2: open, pick
        # case 3: open, place
        elif (a_other_name == "open_other") and (a_ego_name == "pick_ego" or a_ego_name == "place_ego"):
            
            open_door = o.open_door
            obj_regions = o.obj_regions
            holding = o.holding
            
        # case 4: open, close
        # case 7: close, open
        elif (a_other_name == "open_other" and a_ego_name == "close_ego") or (a_other_name == "close_other" and a_ego_name == "open_ego"):
            
            open_door = o.open_door
        
        # case 5: close, pick
        # case 6: close, place
        elif a_other_name == "close_other" and (a_ego_name == "pick_ego" or a_ego_name == "place_ego"):
            
            open_door = o.open_door
            obj_regions = o.obj_regions
            holding = o.holding
        
        
        # others 
        else: # ego: non-deterministic, other: non-deterministic
            
            # ego actions
            if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego": # update NOT taking place!!!
                rob_regions[a_ego_args[0]] = o.rob_regions[a_ego_args[0]]
                
                    
            elif a_ego_name == "pick_ego" or a_ego_name == "place_ego":
                holding[a_ego_args[0]] = o.holding[a_ego_args[0]]
                obj_regions[a_ego_args[1]] = o.obj_regions[a_ego_args[1]]
            
            elif a_ego_name == "open_ego" or a_ego_name == "close_ego": # open or shut door
                open_door = o.open_door
                
            elif a_ego_name == "nothing_ego":
                pass
            
            # other agent's actions     
            if a_other_name == "transit_other" or a_other_name == "transfer_other":
                # assumption: transit is only confusable with transit or noop
                # assumption: transfer is only confusable with transfer or noop
                rob_regions[a.args[0]] = o.rob_regions[a.args[0]]
            
            elif a_other_name == "pick_other" or a_other_name == "place_other":
                # assumption: pick is only confusable with pick or noop
                # assumption: place is only confusable with place or noop
                holding[a.args[0]] = o.holding[a.args[0]]
                obj_regions[a.args[1]] = o.obj_regions[a.args[1]]
            
            elif a_other_name == "open_other" or a_other_name == "close_other":
                # assumption: open is only confusable with open or noop
                # assumption: close is only confusable with close or noop
                open_door = o.open_door
            
            elif a_other_name == "nothing_other":
                # no change in state
                pass
           
        next_actions = o.next_actions
            
        return EnvBelief(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,next_actions=next_actions)

    def abstract(self, store: AliasStore):
        
        ab = []
        
        # true state
        for rob in self.holding.keys():
            ab += [Atom("holding",[rob,obj]) for obj in self.holding[rob]]
        for rob in self.rob_regions.keys():
            ab += [Atom("in_rob",[rob,self.rob_regions[rob]])]
        for obj in self.obj_regions.keys():
            if self.obj_regions[obj] !="":
                ab += [Atom("in_obj",[obj,self.obj_regions[obj]])]
        if self.open_door:
            ab += [Atom("open",[DOOR])]
        
        # next actions
        if self.next_actions != []:
            for next_action in self.next_actions:
                
                name,args = next_action.split("-")
                args=list(args.split("%"))
                
                rob=args[0]
                if Atom("is_ego",[rob]) not in store.certified:
                    ab += [Atom(name,args)]
            
        return AbstractBelief(ab)

    # def vectorize(self):
    #     return np.array([int(obj in self.holding) for obj in OBJECTS])
      
def get_next_actions_execute(a, b, store): # human operator : tedious, kind of works
    
    a_other_name,a_ego_name = a.name.split("*")
    if a_other_name == "transfer_other":
        n_args=4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        n_args = 1
    else:
        n_args = 3
        
    print("ego attempts action ..")
    print(a_ego_name)
    print(a.args[n_args:])
    print("... belief ...")
    b_temp=copy.deepcopy(b) 
    print(b_temp.abstract(store))
    print("applicable actions...")
    
    next_actions=[]
    others = []
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    for rob in others: # one list of outcomes per robot
        
        ab = b_temp.abstract(store)
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        for reg in REGIONS:
            for obj in OBJ_REGIONS.keys():
                if Atom("holding",[rob,obj]) in ab.items: # robot is holding an object: it can transfer or place
                    if Atom("in_rob",[rob,reg]) not in ab.items:
                        applicable_actions_rob.append(Atom("transfer_action",[rob,obj,reg]))
                    else:
                        if Atom("stable",[obj,reg]) in store.certified:
                            if (reg==REGIONS[0] and Atom("open",[DOOR]) in ab.items) or reg!=REGIONS[0]: # region accessibility
                                applicable_actions_rob.append(Atom("place_action",[rob,obj]))
                else: # robot is not holding an object: it can transit, open, pick, depending on where it is and where the objects are
                    if Atom("in_rob",[rob,reg]) in ab.items: # cannot move to reg
                        if reg==REGIONS[1]:
                            if Atom("open",[DOOR]) not in ab.items: 
                                applicable_actions_rob.append(Atom("open_action",[rob]))
                            else:
                                applicable_actions_rob.append(Atom("close_action",[rob]))
                        if Atom("in_obj",[obj,reg]) in ab.items and Atom("in_rob",[rob,reg]):
                            # accessibility of obj
                            if (reg==REGIONS[0] and Atom("open",[DOOR]) in ab.items) or (reg!=REGIONS[0]):
                                applicable_actions_rob.append(Atom("pick_action",[rob,obj]))
                    else: # can move to reg
                        applicable_actions_rob.append(Atom("transit_action",[rob,reg]))
        
        while True:
            
            for i,act in enumerate(applicable_actions_rob):
                print(str(i)+". "+act.pred_name+str(act.args))
                
            choice = input("choose an action \n")
            if int(choice)>=0 and int(choice)<len(applicable_actions_rob):
                break
            else:
                print("invalid choice, enter again")
          
        
        
        observed_action_rob = applicable_actions_rob[int(choice)]   
        print(observed_action_rob)
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name=="transit_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "transfer_action":
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
        elif name == "pick_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "place_action":
            a_other=name+"-"+rob+"%"+args[1]
        else: # open, close, nothing
            a_other=name+"-"+rob
            
        next_actions.append(a_other)
            
    
    return next_actions # for all the agents
def get_next_actions_effects(a, b, store): # human operator : tedious, kind of works
    
    b_temp=copy.deepcopy(b) 
    # b_temp=b_temp.update(a,None,None)
    
    next_actions = []
    
    others = []
    
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    for rob in others: # one list of outcomes per robot
        
        ab = b_temp.abstract(store)
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        
        observed_action_rob = ""

        for reg in REGIONS:
            for obj in OBJ_REGIONS.keys():
                if Atom("holding",[rob,obj]) in ab.items: # robot is holding an object: it can transfer or place
                    if Atom("in_rob",[rob,reg]) not in ab.items:
                        applicable_actions_rob.append(Atom("transfer_action",[rob,obj,reg]))
                    else:
                        if Atom("stable",[obj,reg]) in store.certified:
                            if (reg==REGIONS[0] and Atom("open",[DOOR]) in ab.items) or reg!=REGIONS[0]: # region accessibility
                                applicable_actions_rob.append(Atom("place_action",[rob,obj]))
                                observed_action_rob=Atom("place_action",[rob,obj])
                else: # robot is not holding an object: it can transit, open, pick, depending on where it is and where the objects are
                    if Atom("in_rob",[rob,reg]) in ab.items: # cannot move to reg
                        if reg==REGIONS[1]:
                            if Atom("open",[DOOR]) not in ab.items: # door
                                applicable_actions_rob.append(Atom("open_action",[rob]))
                                observed_action_rob=Atom("open_action",[rob])
                                
                            else:
                                applicable_actions_rob.append(Atom("close_action",[rob]))
                                observed_action_rob=Atom("close_action",[rob])
                                
                        if Atom("in_obj",[obj,reg]) in ab.items and Atom("in_rob",[rob,reg]):
                            # accessibility of obj
                            if (reg==REGIONS[0] and Atom("open",[DOOR]) in ab.items) or (reg!=REGIONS[0]):
                                applicable_actions_rob.append(Atom("pick_action",[rob,obj]))
                                observed_action_rob=Atom("pick_action",[rob,obj])
                                
                    else: # can move to reg
                        applicable_actions_rob.append(Atom("transit_action",[rob,reg]))

        # simulation: if pick, place, open, close are applicable, the other agent tends to perform that action
        if observed_action_rob == "":
            observed_action_rob = random.choice(applicable_actions_rob)
        else: # 70% of time "goal directed" actions, 30% of the time random
            if random.random()<0.3:
                observed_action_rob = random.choice(applicable_actions_rob)
                
        #observed_action_rob = Atom("nothing_action",[rob]) # INACTIVE OTHER AGENT!!!
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name=="transit_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "transfer_action":
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
        elif name == "pick_action":
            a_other=name+"-"+rob+"%"+args[1]
        elif name == "place_action":
            a_other=name+"-"+rob+"%"+args[1]
        else: # open, close, nothing
            a_other=name+"-"+rob
            
        next_actions.append(a_other)
    
    
    return next_actions # for all the agents

# other agents actions
def transit_transfer_other_execute_fn(a, b, s, store):
    
    rob_regions = b.rob_regions.copy()    
    
    rob_regions[a.args[0]] = s.rob_regions[a.args[0]]
    
    return s, EnvObservation(rob_regions=rob_regions)
def transit_transfer_other_effects_fn(a, b, store):
    p = [] # sample probabilities for weighting
    for reg in REGIONS:
        if reg == a.args[2]:
            p.append(0.7) # random outcome
            # p.append(1.0) # deterministic outcome
        else:
            p.append(0.15) # random outcome
            # p.append(0.0) # deterministic outcome
    rob_regions = b.rob_regions.copy()    
    rob_regions[a.args[0]] = random.choice(REGIONS)#np.random.choice(np.array(ROB_REGIONS),p=np.array(p)) # weighted probabilities
    
    return rob_regions
    
def pick_other_execute_fn(a, b, s, store):        
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
            
    if s.obj_regions[a.args[1]] == "" and s.holding[a.args[0]] == [a.args[1]]: # picked
        print("picked")
        obj_regions[a.args[1]] = ""
        holding[a.args[0]] = [a.args[1]]
    else:
        print("not picked")
    
    return s, EnvObservation(obj_regions=obj_regions,holding=holding)
def pick_other_effects_fn(a, b, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    # if random.random()<=1.0: # deterministic outcome
    if random.random()<=0.9: # picked # tweak this number to have a higher probability of pick
        obj_regions[a.args[1]] = ""
        holding[a.args[0]] = [a.args[1]]
    
    return obj_regions,holding
    
def place_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    if s.obj_regions[a.args[1]] == a.args[2] and s.holding[a.args[0]] == []: # placed
        print("placed")
        obj_regions[a.args[1]] = a.args[2]
        holding[a.args[0]] = []
    else:
        print("not placed")
    
    return s, EnvObservation(obj_regions=obj_regions,holding=holding)
def place_other_effects_fn(a, b, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    # if random.random()<=1.0: # deterministic outcome
    if random.random()<=0.9: # placed # tweak this number to have a higher probability of place
        obj_regions[a.args[1]] = a.args[2]
        holding[a.args[0]] = []
    
    return obj_regions,holding
    # o = EnvObservation(obj_regions=obj_regions,holding=holding)
    # new_belief = b.update(a, o, store)
    # return AbstractBeliefSet.from_beliefs([new_belief], store)

def open_other_execute_fn(a, b, s, store):
    
    open_door = b.open_door
    
    if s.open_door: # opened
        print("opened")
        open_door = True
    else:
        print("not opened")
    
    return s, EnvObservation(open_door=open_door)
def open_other_effects_fn(a, b, store):
    
    open_door = b.open_door
    
    # if random.random()<=1.0: # deterministic outcome
    if random.random()<=0.8:
        open_door = True
        
    return open_door    
    # o = EnvObservation(open_door=open_door)
    # new_belief = b.update(a, o, store)
    # return AbstractBeliefSet.from_beliefs([new_belief], store)

def close_other_execute_fn(a, b, s, store):

    open_door = b.open_door
    
    if not s.open_door: # closed
        print("closed")
        open_door = False
    else:
        print("not closed")
    
    return s, EnvObservation(open_door=open_door)
def close_other_effects_fn(a, b, store):
    
    open_door = b.open_door
    
    # if random.random()<=1.0: # deterministic outcome
    if random.random()<=0.8:
        open_door = False
        
    return open_door
    # o = EnvObservation(open_door=open_door)
    # new_belief = b.update(a, o, store)
    # return AbstractBeliefSet.from_beliefs([new_belief], store)    

# joint actions
def joint_execute_fn(a, b, s, store):
    
    holding = b.holding.copy()
    open_door = b.open_door
    rob_regions = b.rob_regions.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name, a_ego_name = a.name.split("*")
        
    if a_other_name == "transfer_other":
        nargs = 4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        nargs = 1
    else:
        nargs = 3
    
    args_ego = a.args[nargs:]
    
    # remove ego's previous action
    for na in s.next_actions: 
        name,args = na.split("-")
        args=args.split("%")
        if args[0] == args_ego[0]:
            s.next_actions.remove(na)
    

    # special cases
    # remember: next actions!!
    # case 1: place, pick
    if a_other_name == "place_other" and a_ego_name == "pick_ego": 
        print("special case")
        print(a.name)
        s,obs = place_other_execute_fn(a, b, s, store)
        obj_regions = s.obj_regions
        if obj_regions[a.args[1]] == a.args[2]: # other agent placed the object
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim: # 90% success
                holding[a.args[3]] = [a.args[1]]
                obj_regions[a.args[1]] = ""
                # update state
                s.holding[a.args[3]] = [a.args[1]]
                s.obj_regions[a.args[1]] = ""
            
    # case 2: open, pick
    elif a_other_name == "open_other" and a_ego_name == "pick_ego":
        
        s,obs = open_other_execute_fn(a, b, s, store)
        open_door = s.open_door
        
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim: # 90% success
                holding[a.args[1]] = [a.args[2]]
                obj_regions[a.args[2]] = ""
                # update state
                s.holding[a.args[1]] = [a.args[2]]
                s.obj_regions[a.args[2]] = ""
        
            
    # case 3: open, place
    elif a_other_name == "open_other" and a_ego_name == "place_ego":

        s,obs = open_other_execute_fn(a, b, s, store)
        open_door = s.open_door
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if random.random()<0.9: # 90% success
                holding[a.args[1]] = []
                obj_regions[a.args[2]] = a.args[3]
                # update state
                s.holding[a.args[1]] = []
                s.obj_regions[a.args[2]] = a.args[3]
        
        
    # case 4: open, close
    elif a_other_name == "open_other" and a_ego_name == "close_ego":

        s,obs = open_other_execute_fn(a, b, s, store)
        open_door = s.open_door
        if open_door and random.random()<CLOSE_EGO: # 90% success
            open_door = False
            # update state
            s.open_door = open_door
            
    # case 5: close, pick
    
    elif a_other_name == "close_other" and a_ego_name == "pick_ego":

        s,obs = close_other_execute_fn(a, b, s, store)
        open_door = obs.open_door
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim: # 90% success
                holding[a.args[1]] = [a.args[2]]
                obj_regions[a.args[2]] = ""
                # update state
                s.holding[a.args[1]] = [a.args[2]]
                s.obj_regions[a.args[2]] = ""
        
    # case 6: close, place
    
    elif a_other_name == "close_other" and a_ego_name == "place_ego":

        s,obs = close_other_execute_fn(a, b, s, store)
        open_door = obs.open_door
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if random.random()<0.9: # 90% success
                holding[a.args[1]] = []
                obj_regions[a.args[2]] = a.args[3]
                # update state
                s.holding[a.args[1]] = []
                s.obj_regions[a.args[2]] = a.args[3]
        
    # case 7: close, open
    elif a_other_name == "close_other" and a_ego_name == "open_ego":

        s,obs = close_other_execute_fn(a, b, s, store)
        open_door = s.open_door
        if not open_door and random.random()<OPEN_EGO: 
            open_door = True
            # update state
            s.open_door = open_door
        
    else: # rest
        
        # other: non-deterministic, ego: deterministic
        
        
        if a_other_name == "transit_other" or a_other_name == "transfer_other":
            s,obs = transit_transfer_other_execute_fn(a, b, s, store)
            rob_regions[a.args[0]] = s.rob_regions[a.args[0]]
             
        elif a_other_name == "open_other":
            s,obs = open_other_execute_fn(a, b, s, store)
            open_door = s.open_door
            
            
        elif a_other_name == "close_other":
            s,obs = close_other_execute_fn(a, b, s, store)
            open_door = s.open_door
            
            
        elif a_other_name == "pick_other":
            s,obs = pick_other_execute_fn(a, b, s, store)
            obj_regions[a.args[1]] = s.obj_regions[a.args[1]]
            holding[a.args[0]] = s.holding[a.args[0]]
            
            
        elif a_other_name == "place_other":
            s,obs = place_other_execute_fn(a, b, s, store)
            obj_regions[a.args[1]] = s.obj_regions[a.args[1]]
            holding[a.args[0]] = s.holding[a.args[0]]
            
        # ego: non-deterministic
        
        if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
            if random.random()<0.9:
                rob_regions[args_ego[0]] = args_ego[2]
                # update state
                s.rob_regions[args_ego[0]] = args_ego[2]
        elif a_ego_name == "pick_ego":
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim:
                holding[args_ego[0]] = [args_ego[1]]
                obj_regions[args_ego[1]] = ""
                # update state
                s.holding[a.args[1]] = [a.args[2]]
                s.obj_regions[a.args[2]] = ""
        elif a_ego_name == "place_ego":
            if random.random()<0.9:
                holding[args_ego[0]] = []
                obj_regions[args_ego[1]] = args_ego[2]
                # update state
                s.holding[args_ego[0]] = []
                s.obj_regions[args_ego[1]] = args_ego[2]
        elif a_ego_name == "open_ego":
            if random.random()<OPEN_EGO:
                open_door = True
                # update state
                s.open_door = True
        elif a_ego_name == "close_ego":
            if random.random()<CLOSE_EGO:
                open_door = False
                # update state
                s.open_door = False
            
    
    # next action of ego
    if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[2]
        # additional argument 
        if a_ego_name == "transfer_ego":
            next_action = next_action + "%" + args_ego[3]
            
    elif a_ego_name == "pick_ego" or a_ego_name == "place_ego":
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1]
        
    else:
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]
        
        
    # add ego's next action for the other agent     
    s.next_actions.append(next_action)   
    
    next_actions = s.next_actions.copy() 
    
    for na in next_actions: # replace other agents' previous action with noop (temporary, till observation is received)
        name,args = na.split("-")
        args=args.split("%")
        if args[0] != args_ego[0]:
            next_actions.remove(na)
            next_actions.append("nothing_action-"+args[0])
    
           
    #  next actions
    
    b_temp = copy.deepcopy(b)
    o = EnvObservation(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions)
    b_temp = b_temp.update(a,o,store)    
    # s is updated by non-deterministic ego execution            
    return s, EnvObservation(holding=s.holding,open_door=s.open_door,rob_regions=s.rob_regions,obj_regions=s.obj_regions,next_actions=next_actions)  
def joint_effects_fn(a, b, store):

    holding = b.holding.copy()
    open_door = b.open_door
    rob_regions = b.rob_regions.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "transfer_other":
        n_args = 4
    elif a_other_name == "open_other" or a_other_name == "close_other" or a_other_name == "nothing_other":
        n_args = 1
    else:
        n_args = 3
    
    args_ego = a.args[n_args:]
    
    
    # special cases
    # remember: next actions!!
    # case 1: place, pick
    if a_other_name == "place_other" and a_ego_name == "pick_ego": 
        
        obj_regions,holding = place_other_effects_fn(a, b, store)
        
        if obj_regions[a.args[1]] == a.args[2]: # other agent placed the object
            if random.random()<0.9: # ego:non-deterministic # use simulator!!
                holding[a.args[3]] = [a.args[1]]
                obj_regions[a.args[1]] = ""
        
            
    # case 2: open, pick
    elif a_other_name == "open_other" and a_ego_name == "pick_ego":
        
        open_door = open_other_effects_fn(a, b, store)
        
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim:
                holding[a.args[1]] = [a.args[2]]
                obj_regions[a.args[2]] = ""
        
            
    # case 3: open, place
    elif a_other_name == "open_other" and a_ego_name == "place_ego":
        
        open_door = open_other_effects_fn(a, b, store)
        
        if (a.args[3] == REGIONS[0] and open_door) or a.args[3] != REGIONS[0]: # in mug and door open or not in mug!
            if random.random()<0.9:
                holding[a.args[1]] = []
                obj_regions[a.args[2]] = a.args[3]
        
            
    # case 4: open, close
    elif a_other_name == "open_other" and a_ego_name == "close_ego":
        
        if random.random()<CLOSE_EGO:
            open_door = False
        
        
    # case 5: close, pick
    # case 6: close, place
    elif a_other_name == "close_other" and (a_ego_name == "pick_ego" or a_ego_name == "place_ego"):
        
        open_door = close_other_effects_fn(a, b, store)
        
        if a_ego_name == "pick_ego":
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim:
                holding[a.args[1]] = [a.args[2]]
                obj_regions[a.args[2]] = ""
        else:
            if random.random()<0.9:
                holding[a.args[1]] = []
                obj_regions[a.args[2]] = a.args[3]
        
    # case 7: close, open
    elif a_other_name == "close_other" and a_ego_name == "open_ego":
        
        if random.random()<OPEN_EGO:
            open_door = True
        
    else: # rest
        # other: non-deterministic
        
        if a_other_name == "transit_other" or a_other_name == "transfer_other":
            
            rob_regions = transit_transfer_other_effects_fn(a, b, store)
            
            
        elif a_other_name == "open_other":
            
            open_door = open_other_effects_fn(a, b, store)
            
        elif a_other_name == "close_other":
            
            open_door = close_other_effects_fn(a, b, store)
            
        elif a_other_name == "pick_other":
            
            obj_regions,holding = pick_other_effects_fn(a, b, store)
            
            
        elif a_other_name == "place_other":
            
            obj_regions,holding = place_other_effects_fn(a, b, store)
            
        # ego: non-deterministic
        
        if a_ego_name == "transit_ego" or a_ego_name == "transfer_ego":
            if random.random()<0.9:
                rob_regions[args_ego[0]] = args_ego[2]
        elif a_ego_name == "pick_ego":
            if a.args[2] == REGIONS[0]: # cabinet
                pick_ego_sim = CABINET_PICK_EGO_SIM
            else:
                pick_ego_sim = SIMPLE_PICK_EGO_SIM
            if random.random()<pick_ego_sim:
                holding[args_ego[0]] = [args_ego[1]]
                obj_regions[args_ego[1]] = ""
        elif a_ego_name == "place_ego":
            if random.random()<0.9:
                holding[args_ego[0]] = []
                obj_regions[args_ego[1]] = args_ego[2]
        elif a_ego_name == "open_ego":
            if random.random()<OPEN_EGO:
                open_door = True
        elif a_ego_name == "close_ego":
            if random.random()<CLOSE_EGO:
                open_door = False
    
    # resulting state
    b_temp = copy.deepcopy(b)
    obs = EnvObservation(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions)
    b_temp = b_temp.update(a,obs,store)   
    next_actions = get_next_actions_effects(a, b_temp, store)
    
    o = EnvObservation(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,next_actions=next_actions)    
    new_belief=b.update(a,o,store)
    return AbstractBeliefSet.from_beliefs([new_belief], store)        
            

# rest of the ego-actions have deterministic effects! 

# Set up environment dynamics
class ToyDiscrete(TampuraEnv):
    
    def initialize(self,ego=f"{ROB}1",s=EnvState()):
        
        self.ego=ego
        
        store = AliasStore()
        
        for rob in ROBOTS:
            
            store.set(rob, rob, "robot")
        # store.set(ego,ego,"robot")
            
        for region in REGIONS:
            store.set(region, region, "region")
        
        store.set(MUG, MUG, "physical")
        store.set(DOOR, DOOR, "door")
        
        store.certified.append(Atom("stable",[MUG,REGIONS[0]]))
        store.certified.append(Atom("stable",[MUG,REGIONS[2]]))
        
        store.certified.append(Atom("is_ego",[ego]))

        holding = s.holding
        open_door = s.open_door
        rob_regions = s.rob_regions
        obj_regions = s.obj_regions
        next_actions = s.next_actions

        b = EnvBelief(holding=holding,open_door=open_door,rob_regions=rob_regions,obj_regions=obj_regions,
                      next_actions=next_actions)

        return b, store

    def get_problem_spec(self) -> ProblemSpec:
        
        actions_other = ACTION_NAMES
        
        others=[]
        for rob in ROBOTS:
            if rob != self.ego:
                others.append(rob)

        predicates = [
            Predicate("is_ego",["robot"]),
            Predicate("holding", ["robot","physical"]),
            Predicate("stable",["physical","region"]),
            Predicate("in_rob",["robot","region"]),
            Predicate("in_obj",["physical","region"]),
            Predicate("open",["door"]),
        ] 
        action_predicates = [Predicate("transit_action",["robot","region"]),Predicate("transfer_action",["robot","physical","region"]),Predicate("pick_action",["robot","physical"]),
                             Predicate("place_action",["robot","physical"]),Predicate("open_action",["robot"]),Predicate("close_action",["robot"]),Predicate("nothing_action",["robot"])]
        
        predicates += action_predicates
        
        possible_outcomes = [[Atom("transit_action",[rob,reg]) for reg in REGIONS]+[Atom("transfer_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                            [Atom("pick_action",[rob,obj]) for obj in OBJ_REGIONS.keys()] + [Atom("place_action",[rob,obj])for obj in OBJ_REGIONS.keys()] +
                            [Atom("open_action",[rob]),Atom("close_action",[rob]),Atom("nothing_action",[rob])] for rob in others]
        
        possible_outcomes_pick_place = [[Atom("transit_action",[rob,reg]) for reg in REGIONS]+[Atom("transfer_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                                        [Atom("pick_action",[rob,obj]) for obj in OBJ_REGIONS.keys()] + [Atom("place_action",[rob,obj])for obj in OBJ_REGIONS.keys()] +
                                        [Atom("nothing_action",[rob])] for rob in others]
        
        possible_outcomes_open_close = [[Atom("transit_action",[rob,reg]) for reg in REGIONS]+
                                        [Atom("open_action",[rob]),Atom("close_action",[rob]),Atom("nothing_action",[rob])] for rob in others]
        
        possible_outcomes_transit = [[Atom("transit_action",[rob,reg]) for reg in REGIONS]+
                                     [Atom("pick_action",[rob,obj]) for obj in OBJ_REGIONS.keys()] +
                                     [Atom("open_action",[rob]),Atom("close_action",[rob]),Atom("nothing_action",[rob])] for rob in others]
        
        possible_outcomes_transfer = [[Atom("transfer_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                                      [Atom("place_action",[rob,obj])for obj in OBJ_REGIONS.keys()] +
                                      [Atom("nothing_action",[rob])] for rob in others]
        
        
        # modify preconditions, effects and execute functions for observation
        action_schemas_ego = [
            
            # ego-agent
            ActionSchema(
                name="pick_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Or([Not(Atom("in_rob",["?rob1",REGIONS[0]])),And([Atom("in_rob",["?rob1",REGIONS[0]]),Atom("open",[DOOR])])]), # TODO: modify!! accesibility of mug: derived predicate
                               Atom("in_obj",["?obj1","?reg1"]), # object is in region from where pick is attempted
                               Atom("in_rob",["?rob1","?reg1"]), # robot is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])], # deterministic

            ),
            
            
            ActionSchema(
                name="place_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Or([Not(Atom("in_rob",["?rob1",REGIONS[0]])),And([Atom("in_rob",["?rob1",REGIONS[0]]),Atom("open",[DOOR])])]), # TODO: modify!! accessibility of region
                               Atom("in_rob",["?rob1","?reg1"]), # robot is in region where place is attempted
                               Atom("holding",["?rob1","?obj1"]), # robot is holding the object that is to be placed 
                               Atom("stable",["?obj1","?reg1"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])], # deterministic 
            ),
            

            ActionSchema(
                name="transit_ego",
                inputs=["?rob1","?reg1","?reg2"],
                input_types=["robot","region","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_rob",["?rob1","?reg1"]),
                               Not(Atom("in_rob",["?rob1","?reg2"])),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob1","?reg1"]),Atom("in_rob",["?rob1","?reg2"])])],
            ),
            ActionSchema(
                name="transfer_ego",
                inputs=["?rob1","?reg1","?reg2","?obj1"],
                input_types=["robot","region","region","physical"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_rob",["?rob1","?reg1"]),
                               Not(Atom("in_rob",["?rob1","?reg2"])),
                               Atom("holding",["?rob1","?obj1"])],
                verify_effects=[OneOf([Atom("in_rob",["?rob1","?reg1"]),Atom("in_rob",["?rob1","?reg2"])])],
            ),
            ActionSchema(
                name="open_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Not(Atom("open",[DOOR])),
                               Atom("in_rob",["?rob1",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Atom("open",[DOOR])],
                
            ),
            ActionSchema(
                name="close_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("open",[DOOR]),
                               Atom("in_rob",["?rob1",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Not(Atom("open",[DOOR]))],
        
            ),
            
            ActionSchema(
                name="nothing_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"])],
                effects=[],
            ),
        ]
        
        action_schemas_other = [
            
            # other agents
            ActionSchema(
                name="pick_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("pick_action",["?rob2","?obj2"]), # other agents' turn
                               Or([Not(Atom("in_rob",["?rob2",REGIONS[0]])),And([Atom("in_rob",["?rob2",REGIONS[0]]),Atom("open",[DOOR])])]), # accesibility of mug: derived predicate
                               Atom("in_obj",["?obj2","?reg3"]), # object is in region from where pick is attempted
                               Atom("in_rob",["?rob2","?reg3"]), # robot is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])]+[OneOf(po) for po in possible_outcomes_pick_place],
            ),
            
            
            ActionSchema(
                name="place_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("place_action",["?rob2","?obj2"]), # other agents' turn
                               Or([Not(Atom("in_rob",["?rob2",REGIONS[0]])),And([Atom("in_rob",["?rob2",REGIONS[0]]),Atom("open",[DOOR])])]), # accessibility of region
                               Not(Atom("in_obj",["?obj2","?reg3"])), # object is in region where place is attempted
                               Atom("in_rob",["?rob2","?reg3"]), # robot is in region where place is attempted
                               Atom("holding",["?rob2","?obj2"]), # robot is holding the object that is to be placed 
                               Atom("stable",["?obj2","?reg3"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])]+[OneOf(po) for po in possible_outcomes_pick_place],
            ),
            
            ActionSchema(
                name="transit_other",
                inputs=["?rob2","?reg3","?reg4"],
                input_types=["robot","region","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("transit_action",["?rob2","?reg4"]), # other agents' turn
                               Atom("in_rob",["?rob2","?reg3"]),
                               Not(Atom("in_rob",["?rob2","?reg4"])),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob2",reg]) for reg in REGIONS])]+[OneOf(po) for po in possible_outcomes_transit],
            ),
            
            ActionSchema(
                name="transfer_other",
                inputs=["?rob2","?reg3","?reg4","?obj2"],
                input_types=["robot","region","region","physical"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("transfer_action",["?rob2","?obj2","?reg4"]), # other agents' turn
                               Atom("in_rob",["?rob2","?reg3"]),
                               Not(Atom("in_rob",["?rob2","?reg4"])),
                               Atom("holding",["?rob2","?obj2"]),
                               ],
                verify_effects=[OneOf([Atom("in_rob",["?rob2",reg]) for reg in REGIONS])]+[OneOf(po) for po in possible_outcomes_transfer],
            ),
            ActionSchema(
                name="open_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("open_action",["?rob2"]), # other agents' turn
                               Not(Atom("open",[DOOR])),
                               Atom("in_rob",["?rob2",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Atom("open",[DOOR])]+[OneOf(po) for po in possible_outcomes_open_close], # TODO: modify     
            ),
            ActionSchema(
                name="close_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("close_action",["?rob2"]), # other agents' turn
                               Atom("open",[DOOR]),
                               Atom("in_rob",["?rob2",REGIONS[1]]),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"]))],
                verify_effects=[Not(Atom("open",[DOOR]))]+[OneOf(po) for po in possible_outcomes_open_close], # TODO: modify
            ),
            ActionSchema(
                name="nothing_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])),
                               Atom("nothing_action",["?rob2"])],
                verify_effects=[OneOf(po) for po in possible_outcomes],
            )
            
            
        ]
        
        
        
        action_schemas = []
        
        for as_other in action_schemas_other:
            
            as_other_name = as_other.name
            
            for as_ego in action_schemas_ego:
                
                as_ego_name = as_ego.name
                schema = ActionSchema()
                
                if (as_other_name == "transfer_other" and (as_ego_name == "transfer_ego" or as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                    (as_other_name == "pick_other" and (as_ego_name == "transfer_ego" or as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                        (as_other_name == "open_other" and as_ego_name == "open_ego") or (as_other_name == "close_other" and as_ego_name == "close_ego") or \
                            (as_other_name == "place_other" and (as_ego_name == "place_ego" or as_ego_name == "transfer_ego")): # not possible under beliefs
                    
                    continue
                
                # special cases
                # assumption: other agent acts before ego agent
                # assumption: pick is confusible with place in the sense nothing happens and vice versa
                # transit, transfer regions are confusible, nothing may happen (same region)
                # open, close are confusible with each other in the sense nothing happens
                # noop observation is deterministic
                
                # case 1: place, pick
                elif as_other_name == "place_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = ["?rob2","?obj1","?reg1","?rob1"]
                    schema.input_types = ["robot","physical","region","robot"]
                    schema.preconditions = [Atom("place_action",["?rob2","?obj1"]),Atom("is_ego",["?rob1"]),Not(Atom("is_ego",["?rob2"])),Atom("holding",["?rob2","?obj1"]),
                                            Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])),Atom("in_rob",["?rob1","?reg1"]),
                                            Atom("in_rob",["?rob2","?reg1"]),Atom("stable",["?obj1","?reg1"]),
                                            Or([Not(Atom("in_rob",["?rob2",REGIONS[0]])),And([Atom("in_rob",["?rob2",REGIONS[0]]),Atom("open",[DOOR])])]), # accessibility of region for place
                                            ]
                    schema.effects = [Not(Atom("place_action",["?rob2","?obj1"]))]
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes_pick_place] + [OneOf([Atom("holding",["?rob1","?obj1"]),Atom("holding",["?rob2","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])]
                    
                # case 2: open, pick
                elif as_other_name == "open_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("in_obj",["?obj1","?reg1"]), Atom("in_rob",["?rob1","?reg1"]), 
                                                                     Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"]))]
                    schema.effects = as_other.effects 
                    schema.verify_effects = as_other.verify_effects + [OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])]
                    
                    
                # case 3: open, place
                elif as_other_name == "open_other" and as_ego_name == "place_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("holding",["?rob1","?obj1"]), Atom("in_rob",["?rob1","?reg1"]), 
                                                                     Atom("stable",["?obj1","?reg1"])]
                    schema.effects = as_other.effects
                    schema.verify_effects = as_other.verify_effects + [OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])]
                    
                # case 4: open, close
                elif as_other_name == "open_other" and as_ego_name == "close_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes_open_close] + as_ego.verify_effects 
                    
                # case 5: close, pick
                elif as_other_name == "close_other" and as_ego_name == "pick_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Not(Atom("in_obj",["?obj1",REGIONS[0]])),
                                                                     Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), Atom("in_rob",["?rob1","?reg1"])]
                    schema.effects = as_other.effects + as_ego.effects # guaranteed pick in region stable but belief inhibits attempting place in region mug
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects
                    
                # case 6: close, place
                elif as_other_name == "close_other" and as_ego_name == "place_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + [Atom("is_ego",["?rob1"]), Atom("holding",["?rob1","?obj1"]), Atom("in_rob",["?rob1","?reg1"]), 
                                                                     Not(Atom("in_rob",["?rob1",REGIONS[0]])), Atom("stable",["?obj1","?reg1"])]
                    schema.effects = as_other.effects + as_ego.effects # guaranteed place in region stable but belief inhibits attempting place in region mug
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects
                    
                # case 7: close, open
                elif as_other_name == "close_other" and as_ego_name == "open_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = [OneOf(po) for po in possible_outcomes_open_close] + as_ego.verify_effects
                
                # regular cases
                else: 
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects   
                    
                schema.execute_fn = joint_execute_fn
                schema.effects_fn = joint_effects_fn

                action_schemas.append(schema)
                    
       
        
        reward = GOAL

        spec = ProblemSpec(
            predicates=predicates,
            action_schemas=action_schemas,
            reward=reward,
        )

        return spec

# Planner

def main():
    
     ############################  TAMPURA  ###################################
        # Initialize environment
    cfg = {}

    # Set some print options to print out abstract belief, action, observation, and reward
    cfg["task"] = "class_uncertain"
    cfg["planner"] = "tampura_policy"
    cfg["global_seed"] = 0
    cfg["vis"] = False
    cfg["flat_width"] = 1
    cfg["pwa"] = 0.2
    cfg["pwk"] = 3.0
    cfg["envelope_threshold"] = 0.015
    cfg["gamma"] = 0.95
    cfg["decision_strategy"] = "prob"
    cfg["learning_strategy"] = "bayes_optimistic"
    cfg["load"] = "null"
    cfg["real_camera"] = False
    cfg["real_execute"] = False
    cfg["symk_selection"] = "unordered"
    cfg["symk_direction"] = "fw"
    cfg["symk_simple"] = True
    cfg["from_scratch"] = True
    cfg["flat_sample"] = True # disable progressive widening

    # Set some print options to print out abstract belief, action, observation, and reward
    cfg["print_options"] = "ab,a,o,r"
    cfg["vis_graph"] = True
    # batch size 100, num samples 500 num skeletons 100 works best!!
    cfg["batch_size"] = 500 #100 
    cfg["num_samples"] = 500 #500
    # overcome optimism about itself!
    cfg["batch_size"] = 100 #100 
    cfg["num_samples"] = 1000 #500
    
    cfg["max_steps"] = 15
    cfg["num_skeletons"] = 100
    cfg["flat_sample"] = False # TODO: check; may cause progressive widening
    cfg['save_dir'] = os.getcwd()+"/runs/run{}".format(time.time())

    # cfg['from_scratch'] = False # imp: re-use!!! but graph gets too big
    cfg['envelope_threshold'] = 0.05 # enforce reuse

    # state
    s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},open_door=False,
                rob_regions={ROBOTS[0]:REGIONS[-1],ROBOTS[1]:REGIONS[-1]},
                obj_regions={MUG:REGIONS[0]},
                next_actions=["nothing_action-"+ROBOTS[0],"nothing_action-"+ROBOTS[1]])
    # for robot1
    # Initialize 
    env = ToyDiscrete(config=cfg)
    b0, store= env.initialize(ego=ROBOTS[0],s=s)


    # Set up logger to print info
    setup_logger(cfg["save_dir"], logging.INFO)

    # Initialize the policy
    planner = TampuraPolicy(config = cfg, problem_spec = env.problem_spec)

    env.state = copy.deepcopy(s)



    b=b0


    assert env.problem_spec.verify(store)

    save_config(planner.config, planner.config["save_dir"])

    history = RolloutHistory(planner.config)

    st = time.time()
    for step in range(100):

        # robot 1 acts
        # s = copy.deepcopy(env.state)
        env.state = copy.deepcopy(s)
        b.next_actions = s.next_actions # important!!
        a_b = b.abstract(store)
        reward = env.problem_spec.get_reward(a_b, store)
        if reward:
            print("goal achieved")
            break
        
        logging.info("\n" + ("=" * 10) + "t=" + str(step) + ("=" * 10))
        if "s" in planner.print_options:
            logging.info("State: " + str(s))
        if "b" in planner.print_options:
            logging.info("Belief: " + str(b))
        if "ab" in planner.print_options:
            logging.info("Abstract Belief: " + str(a_b))
        if "r" in planner.print_options:
            logging.info("Reward: " + str(reward))
        
        
        action, info, store = planner.get_action(b, store) # should only call effects functions!!??
        
        
        if "a" in planner.print_options:
            logging.info("Action: " + str(action))

        if action.name == "no-op":
            bp = copy.deepcopy(b)
            observation = None
        else:
            observation= env.step(action, b, store) # should call execute function
            bp = b.update(action, observation, store)

            if planner.config["vis"]:
                env.vis_updated_belief(bp, store)

        a_bp = bp.abstract(store)
        history.add(s, b, a_b, action, observation, reward, info, store, time.time() - st)

        reward = env.problem_spec.get_reward(a_bp, store)
        
        if "o" in planner.print_options:
            logging.info("Observation: " + str(observation))
        if "sp" in planner.print_options:
            logging.info("Next State: " + str(env.state))
        if "bp" in planner.print_options:
            logging.info("Next Belief: " + str(bp))
        if "abp" in planner.print_options:
            logging.info("Next Abstract Belief: " + str(a_bp))
        if "rp" in planner.print_options:
            logging.info("Next Reward: " + str(reward))

        # update the belief
        b = bp
        
        s = copy.deepcopy(env.state)
        # human operator:
        
        # apparent action
        
        # remove previous action
        for ac in s.next_actions:
            name,args = ac.split("-")
            args=args.split("%")
            if args[0] == ROBOTS[1]:
                s.next_actions.remove(ac)
                
        
        # get current action
        next_actions = get_next_actions_execute(action,b,store)
        for ac in next_actions:
            name,args = ac.split("-")
            args=args.split("%")
            if args[0] == ROBOTS[1]:
                s.next_actions.append(ac)
                if name == "transit_action" or name == "transfer_action":
                    while True:
                        print(ac)
                        print("True region of movement?")
                        for i,reg in enumerate(REGIONS):
                            print(str(i)+". "+reg)
                        choice = input("Pick region")
                        if int(choice)>=0 and int(choice)<len(REGIONS):
                            break
                    s.rob_regions[args[0]] = REGIONS[int(choice)]
                    print(s.rob_regions[args[0]])
                elif name == "open_action":
                    while True:
                        print(ac)
                        print("Was the door opened?")
                        choice = input("0:No / 1:Yes")
                        if int(choice)==0 or int(choice)==1:
                            break
                    if int(choice)==1:
                        s.open_door = True
                        print("opened")
                elif name == "close_action":
                    while True:
                        print(ac)
                        print("Was the door closed?")
                        choice = input("0:No / 1:Yes")
                        if int(choice)==0 or int(choice)==1:
                            break
                    if int(choice)==1:
                        s.open_door = False
                        print("closed")
                elif name == "pick_action":
                    while True:
                        print(ac)
                        print("Was pick executed?")
                        choice = input("0:No / 1:Yes")
                        if int(choice)==0 or int(choice)==1:
                            break
                    if int(choice) == 1: 
                        s.holding[args[0]] = [args[1]]
                        s.obj_regions[args[1]] = ""
                    print(s.holding[args[0]])
                    print(s.obj_regions)
                elif name == "place_action":
                    while True:
                        print(ac)
                        print("Was place executed?")
                        choice = input("0:No / 1:Yes")
                        if int(choice)==0 or int(choice)==1:
                            break
                    if int(choice) == 1: 
                        s.holding[args[0]] = []
                        s.obj_regions[args[1]] = s.rob_regions[args[0]]
                    print(s.holding[args[0]])
                    print(s.obj_regions)
                        
                        
                break
    
        # true outcome
        
    history.add(env.state, bp, a_bp, None, None, reward, info, store, time.time() - st)
        
    logging.info("=" * 20)

    env.wrapup()

    if not planner.config["real_execute"]:
        save_run_data(history, planner.config["save_dir"])
                
                
                



if __name__ == "__main__":
    # run the main function
    main()
    