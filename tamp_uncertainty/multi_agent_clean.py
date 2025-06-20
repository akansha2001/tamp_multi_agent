
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
from tampura.symbolic import OBJ, Atom, ForAll, Not, Exists, Or, And, OneOf, Eq, eval_expr
from tampura.policies.tampura_policy import TampuraPolicy
from tampura.config.config import load_config, setup_logger

# TODO: different training and execution scenarios, study the MDPs
# 0: human, 1: random, 2: inactive
TRAIN = 1
# 0: human, 1: random, 2: inactive, 3: nominal
EXEC = 1

from pick_successes import SIMPLE_PICK_EGO_SIM, CABINET_PICK_EGO_SIM

ROB = "robot_"
REG = "region_"
MUG = "mug"
REGIONS = [f"{REG}{MUG}",f"{REG}stable_mug"]
ACTION_NAMES = ["pick_action","place_action","clean_action","nothing_action"]

# problem specification: try with just one robot to demonstrate how overall cost increases
ROBOTS=[f"{ROB}1",f"{ROB}2"]
OBJ_REGIONS={MUG:REGIONS[0]}

# higher num_samples needed to learn true transition model

# Test 
# GOAL = Atom("holding",[ROBOTS[0],MUG])
GOAL = And([Atom("clean",[REGIONS[0]]),Atom("in_obj",[MUG,REGIONS[0]])])

# State of the environment
@dataclass
class EnvState(State):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    clean: List[str] = field(default_factory=lambda: [])
    next_actions: List[str] = field(default_factory=lambda: [])
    
# Observation space
@dataclass
class EnvObservation(Observation):
    holding: Dict[str,List[str]] = field(default_factory=lambda: {})
    obj_regions: Dict[str,str] = field(default_factory=lambda:{})
    clean: List[str] = field(default_factory=lambda: [])
    next_actions: List[str] = field(default_factory=lambda: [])

# Belief space
class EnvBelief(Belief):
    def __init__(self, holding={},obj_regions={},clean=[],next_actions=[]):
        # true state
        self.holding = holding
        self.obj_regions = obj_regions
        self.clean = clean
        self.next_actions = next_actions
        

    def update(self, a, o, s):
        
        # dictionary mutations are IN-PLACE!!! use .copy()!!
        holding = self.holding.copy() 
        obj_regions = self.obj_regions.copy()
        clean = self.clean
        next_actions = self.next_actions
        
        
        # get argument index for ego agent
        
        a_other_name,a_ego_name = a.name.split("*")
        
        if a_other_name == "clean_other": # robot, region
            nargs_other = 2
        elif a_other_name == "nothing_other":
            nargs_other = 1
        else: # robot, object, region
            nargs_other = 3
            
        
       
        
        
        if a_other_name == "pick_other" or a_other_name == "place_other":
    
            holding = o.holding
            obj_regions = o.obj_regions
        
        elif a_other_name == "clean_other":
            
            clean = o.clean
        
        if a_ego_name == "pick_ego" or a_ego_name == "place_ego":
            
            holding = o.holding
            obj_regions = o.obj_regions
        
        elif a_ego_name == "clean_ego":
            
            clean = o.clean
            
            
           
        next_actions = o.next_actions
            
        return EnvBelief(holding=holding,clean=clean,obj_regions=obj_regions,next_actions=next_actions)

    def abstract(self, store: AliasStore):
        
        ab = []
        
        # true state
        for rob in self.holding.keys():
            ab += [Atom("holding",[rob,obj]) for obj in self.holding[rob]]
        for obj in self.obj_regions.keys():
            if self.obj_regions[obj] !="":
                ab += [Atom("in_obj",[obj,self.obj_regions[obj]])]
        for clean_region in self.clean:
            ab += [Atom("clean",[clean_region])]
        # next actions
        if self.next_actions != []:
            for next_action in self.next_actions:
                
                name,args = next_action.split("-")
                args=list(args.split("%"))
                
                rob=args[0]
                if Atom("is_ego",[rob]) not in store.certified: # not the ego agent
                    ab += [Atom(name,args)]
            
        return AbstractBelief(ab)

    # def vectorize(self):
    #     return np.array([int(obj in self.holding) for obj in OBJECTS])
      

def get_next_actions_execute(a, b, store): 
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        n_args = 2
    elif a_other_name == "nothing_other":
        n_args = 1
    else: # robot, object, region
        n_args = 3
    
    if EXEC == 0: # human  
        print("ego attempts action ..")
        print(a_ego_name)
        print(a.args[n_args:])
    
    
    next_actions=[]
    others = []
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    # depending on whether or not ego succeeded, different actions are applicable
    # however other agent does not know whether ego succeeded, so all actions should technically be applicable!
    # pruned out by preconditions! 
    for rob in others: # one list of outcomes per robot
        
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("clean_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                applicable_actions_rob.append(Atom("pick_action",[rob,obj,reg]))
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))
        
        if EXEC == 0: # human        
        
            while True:
                
                for i,act in enumerate(applicable_actions_rob):
                    print(str(i)+". "+act.pred_name+str(act.args))
                    
                choice = input("choose an action \n")
                if int(choice)>=0 and int(choice)<len(applicable_actions_rob):
                    break
                else:
                    print("invalid choice, enter again")
            
            observed_action_rob = applicable_actions_rob[int(choice)] 
            
        elif EXEC == 1: # random 
            
            observed_action_rob = random.choice(applicable_actions_rob)
            
        elif EXEC == 2: #inactive
            
            observed_action_rob = Atom("nothing_action",[rob])
            
          
        print(observed_action_rob)
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name == "nothing_action":
            a_other=name+"-"+rob
        elif name == "clean_action": 
            a_other=name+"-"+rob+"%"+args[1]
        else: # pick and place
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
            
        next_actions.append(a_other)
    
    return next_actions # for all the agents
def get_next_actions_effects(a, b, store): # human operator : tedious, kind of works
    
    
    next_actions = []
    
    others = []
    
    for entity in store.als_type:
        if store.als_type[entity]=="robot":
            if Atom("is_ego",[entity]) not in store.certified:
                others.append(entity)
    
    for rob in others: # one list of outcomes per robot
        
        applicable_actions_rob=[]
        # nothing is always applicable
        applicable_actions_rob.append(Atom("nothing_action",[rob]))
        
        for reg in REGIONS:
            
            applicable_actions_rob.append(Atom("clean_action",[rob,reg]))
            
            for obj in OBJ_REGIONS.keys():
                applicable_actions_rob.append(Atom("pick_action",[rob,obj,reg]))
                applicable_actions_rob.append(Atom("place_action",[rob,obj,reg]))

        if TRAIN == 0: # human 
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
        elif TRAIN == 1: # random 
            observed_action_rob = random.choice(applicable_actions_rob)
        elif TRAIN == 2: # inactive agent
            observed_action_rob = Atom("nothing_action",[rob])
        
        name=observed_action_rob.pred_name
        args=observed_action_rob.args
        
        if name == "nothing_action":
            a_other=name+"-"+rob
        elif name == "clean_action": 
            a_other=name+"-"+rob+"%"+args[1]
        else: # pick and place
            a_other=name+"-"+rob+"%"+args[1]+"%"+args[2]
            
        next_actions.append(a_other)
    
    
    return next_actions # for all the agents

# other agents actions
    
def pick_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    if EXEC == 0: # human
        while True:
            print("Was pick executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        picked = int(choice) == 1
    
    elif EXEC == 1: # random
        
        picked = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if picked: 
            s.holding[a.args[0]] = [a.args[1]]
            s.obj_regions[a.args[1]] = ""

            
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
    
    if TRAIN == 0: # human
        while True:
            print("Was pick executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        picked = int(choice)==1
        print(picked)
    elif TRAIN == 1: # random
        picked = random.random()<=0.9
                            
    if picked: 
        obj_regions[a.args[1]] = ""
        holding[a.args[0]] = [a.args[1]]
    
    return obj_regions,holding
    
def place_other_execute_fn(a, b, s, store):
    
    obj_regions = b.obj_regions.copy()
    holding = b.holding.copy()
    
    
    if EXEC == 0: # human
        while True:
            print("Was place executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        placed = int(choice) == 1
    
    elif EXEC == 1: # random
        
        placed = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if placed: 
            s.holding[a.args[0]] = []
            s.obj_regions[a.args[1]] = a.args[2]

    
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
    
    if TRAIN == 0: # human
        while True:
            print("Was place executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        placed = int(choice)==1
        print(placed)
    elif TRAIN == 1: # random
        placed = random.random()<=0.9
    
    if placed: 
        obj_regions[a.args[1]] = a.args[2]
        holding[a.args[0]] = []
    
    return obj_regions,holding
    
def clean_other_execute_fn(a, b, s, store):
    
    clean = b.clean.copy()
    
    if EXEC == 0: # human
        while True:
            print("Was clean executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        cleaned = int(choice) == 1
    
    elif EXEC == 1: # random
        
        cleaned = random.random()<0.9
    
    
        
    if EXEC ==0 or EXEC == 1: # human or random; no change to state for nominal!
        if cleaned: 
            s.clean.append(a.args[1])

    if a.args[1] in s.clean:
        print("cleaned")
        clean.append(a.args[1])  
    else:
        print("not cleaned")  
    
    
    return s, EnvObservation(clean=clean)
def clean_other_effects_fn(a, b, store):
    
    clean = b.clean.copy()
    
    if TRAIN == 0: # human
        while True:
            print("Was clean executed?")
            choice = input("0:No / 1:Yes")
            if int(choice)==0 or int(choice)==1:
                break
        cleaned = int(choice)==1
        print(cleaned)
    elif TRAIN == 1: # random
        cleaned = random.random()<=0.9
    
    if cleaned: 
        clean.append(a.args[1])
    
    return clean

# joint actions
def joint_execute_fn(a, b, s, store):
    
    holding = b.holding.copy()
    clean = b.clean.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        nargs = 2
    elif a_other_name == "nothing_other":
        nargs = 1
    else: # robot, object, region
        nargs = 3
    
    args_ego = a.args[nargs:]
    
    # remove ego's previous action
    for na in s.next_actions: 
        name,args = na.split("-")
        args=args.split("%")
        if args[0] == args_ego[0]:
            s.next_actions.remove(na)
    
    
        
    if a_other_name == "pick_other":
        s,obs = pick_other_execute_fn(a, b, s, store)
    elif a_other_name == "place_other":
        s,obs = place_other_execute_fn(a, b, s, store)
    elif a_other_name == "clean_other":
        s,obs = clean_other_execute_fn(a, b, s, store)
        
    if a_ego_name == "pick_ego":
        if s.obj_regions[args_ego[1]] == args_ego[2]: 
            if random.random()<0.9:
                s.holding[args_ego[0]] = [args_ego[1]]
                s.obj_regions[args_ego[1]] = ""
    elif a_ego_name == "place_ego":
        if random.random()<0.9:
            s.holding[args_ego[0]]=[]
            s.obj_regions[args_ego[1]]=args_ego[2]
    elif a_ego_name == "clean_ego":
        if args_ego[1] not in obj_regions.values():
            if random.random()<0.9:
                s.clean.append(args_ego[1])
    
    
    # next action of ego
    if a_ego_name == "pick_ego" or a_ego_name == "place_ego":
        next_action = a_ego_name[:-3]+"action"+"-"+args_ego[0]+"%"+args_ego[1]+"%"+args_ego[2]
    elif a_ego_name=="clean_ego":
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
    o = EnvObservation(holding=holding,clean=clean,obj_regions=obj_regions)
    b_temp = b_temp.update(a,o,store)    
    # s is updated by non-deterministic ego execution            
    return s, EnvObservation(holding=s.holding,clean=s.clean,obj_regions=s.obj_regions,next_actions=next_actions)  
def joint_effects_fn(a, b, store):

    holding = b.holding.copy()
    clean = b.clean.copy()
    obj_regions = b.obj_regions.copy()
    next_actions = b.next_actions.copy()
    
    a_other_name,a_ego_name = a.name.split("*")
    
    if a_other_name == "clean_other": # robot, region
        nargs = 2
    elif a_other_name == "nothing_other":
        nargs = 1
    else: # robot, object, region
        nargs = 3
    
    args_ego = a.args[nargs:]
 
    
    if a_other_name == "pick_other":
        obj_regions,holding = pick_other_effects_fn(a, b, store)
    elif a_other_name == "place_other": 
        obj_regions,holding = place_other_effects_fn(a, b, store)
    elif a_other_name == "clean_other":
        clean = clean_other_effects_fn(a, b, store)
        

    if a_ego_name == "pick_ego":
        if obj_regions[args_ego[1]] == args_ego[2]: 
            if random.random()<0.9:
                holding[args_ego[0]] = [args_ego[1]]
                obj_regions[args_ego[1]] = ""
    elif a_ego_name == "place_ego":
        if random.random()<0.9:
            holding[args_ego[0]]=[]
            obj_regions[args_ego[1]]=args_ego[2]
    elif a_ego_name == "clean_ego":
        if args_ego[1] not in obj_regions.values():
            if random.random()<0.9:
                clean.append(args_ego[1])
    
    
    # resulting state
    b_temp = copy.deepcopy(b)
    obs = EnvObservation(holding=holding,clean=clean,obj_regions=obj_regions)
    b_temp = b_temp.update(a,obs,store)   
    next_actions = get_next_actions_effects(a, b_temp, store)
    
    o = EnvObservation(holding=holding,clean=clean,obj_regions=obj_regions,next_actions=next_actions)    
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
        
        store.certified.append(Atom("stable",[MUG,REGIONS[0]]))
        store.certified.append(Atom("stable",[MUG,REGIONS[1]]))
        
        store.certified.append(Atom("is_ego",[ego]))

        holding = s.holding
        obj_regions = s.obj_regions
        next_actions = s.next_actions
        clean = s.clean

        b = EnvBelief(holding=holding,clean=clean,obj_regions=obj_regions,
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
            Predicate("in_obj",["physical","region"]),
            Predicate("clean",["region"]),
        ] 
        action_predicates = [Predicate("pick_action",["robot","physical","region"]),
                             Predicate("place_action",["robot","physical","region"]),
                             Predicate("clean_action",["robot","region"]),
                             Predicate("nothing_action",["robot"])]
        
        predicates += action_predicates
        
        possible_outcomes = [[Atom("clean_action",[rob,reg]) for reg in REGIONS]+
                            [Atom("pick_action",[rob,obj,reg]) for obj in OBJ_REGIONS.keys() for reg in REGIONS] + 
                            [Atom("place_action",[rob,obj,reg])for obj in OBJ_REGIONS.keys() for reg in REGIONS] +
                            [Atom("nothing_action",[rob])] for rob in others]
        
        

        
        # modify preconditions, effects and execute functions for observation
        action_schemas_ego = [
            
            # ego-agent
            ActionSchema(
                name="pick_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("in_obj",["?obj1","?reg1"]), # object is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])], 
            ),
            
            
            ActionSchema(
                name="place_ego",
                inputs=["?rob1","?obj1","?reg1"],
                input_types=["robot","physical","region"],
                preconditions=[Atom("is_ego",["?rob1"]), # is the ego agent
                               Atom("holding",["?rob1","?obj1"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj1","?reg1"])),
                               Atom("stable",["?obj1","?reg1"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob1","?obj1"]),Atom("in_obj",["?obj1","?reg1"])])],  
            ),
            
            ActionSchema(
                name="clean_ego",
                inputs=["?rob1","?reg1"],
                input_types=["robot","region"],
                preconditions=[Atom("is_ego",["?rob1"]),
                               Not(Exists(Atom("in_obj",["?obj","?reg1"]),["?obj"],["physical"])), # region is free
                               Not(Atom("clean",["?reg1"])), # region is unclean
                               Not(Exists(Atom("holding",["?rob1","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[Atom("clean",["?reg1"])]
            ),
            
            ActionSchema(
                name="nothing_ego",
                inputs=["?rob1"],
                input_types=["robot"],
                preconditions=[Atom("is_ego",["?rob1"])],
            ),
        ]
        
        action_schemas_other = [
            
            # other agents
            ActionSchema(
                name="pick_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("pick_action",["?rob2","?obj2","?reg3"]), # other agents' turn
                               Atom("in_obj",["?obj2","?reg3"]), # object is in region from where pick is attempted
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])],
            ),
            
            
            ActionSchema(
                name="place_other",
                inputs=["?rob2","?obj2","?reg3"],
                input_types=["robot","physical","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("place_action",["?rob2","?obj2","?reg3"]), # other agents' turn
                               Not(Atom("in_obj",["?obj2","?reg3"])), # object is in region where place is attempted
                               Atom("holding",["?rob2","?obj2"]), # robot is holding the object that is to be placed 
                               Not(Atom("in_obj",["?obj2","?reg3"])),
                               Atom("stable",["?obj2","?reg3"]), # region where place is attempted is stable
                               ],
                verify_effects=[OneOf([Atom("holding",["?rob2","?obj2"]),Atom("in_obj",["?obj2","?reg3"])])],
            ),
            
            ActionSchema(
                name="clean_other",
                inputs=["?rob2","?reg3"],
                input_types=["robot","region"],
                preconditions=[Not(Atom("is_ego",["?rob2"])), # is not the ego agent
                               Atom("clean_action",["?rob2","?reg3"]), # other agents' turn
                               Not(Atom("clean",["?reg3"])),
                               Not(Exists(Atom("in_obj",["?obj","?reg3"]),["?obj"],["physical"])),
                               Not(Exists(Atom("holding",["?rob2","?obj"]),["?obj"],["physical"])), # robot hand is free
                               ],
                verify_effects=[Atom("clean",["?reg3"])],
            ),
            
            ActionSchema(
                name="nothing_other",
                inputs=["?rob2"],
                input_types=["robot"],
                preconditions=[Not(Atom("is_ego",["?rob2"])),
                               Atom("nothing_action",["?rob2"])],
                verify_effects=[],
            )
            
            
        ]
        
        
        
        action_schemas = []
        
        for as_other in action_schemas_other:
            
            as_other_name = as_other.name
            
            for as_ego in action_schemas_ego:
                
                as_ego_name = as_ego.name
                schema = ActionSchema()
                
                # prohibited
                if (as_other_name == "pick_other" and (as_ego_name == "pick_ego" or as_ego_name == "place_ego")) or \
                    (as_other_name == "place_other" and as_ego_name == "place_ego" ) or \
                        (as_other_name == "clean_other" and as_ego_name == "clean_ego") : # not possible under beliefs
                    
                    continue
                
                # special cases
                # assumption: other agent acts before ego agent
                # place_other,clean_ego: works only if other attempts place in region for which clean is not attempted by ego
                elif as_other_name == "place_other" and as_ego_name == "clean_ego":
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions + [Not(Eq("?reg1","?reg3"))] # other does not try to place in region ego is trying to clean
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects + [OneOf(po) for po in possible_outcomes]
                    
                # regular cases
                else: 
                    
                    schema.name = as_other_name+"*"+as_ego_name
                    schema.inputs = as_other.inputs + as_ego.inputs
                    schema.input_types = as_other.input_types + as_ego.input_types
                    schema.preconditions = as_other.preconditions + as_ego.preconditions
                    schema.effects = as_other.effects + as_ego.effects
                    schema.verify_effects = as_other.verify_effects + as_ego.verify_effects + [OneOf(po) for po in possible_outcomes]
                    
                schema.execute_fn = joint_execute_fn
                schema.effects_fn = joint_effects_fn

                action_schemas.append(schema)
                    
        # print([a.name+" "+str(a.inputs) for a in action_schemas])
        
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
    # overcome optimism for inactive agent
    # cfg["batch_size"] = 100 #100 
    # cfg["num_samples"] = 5000 #500
    
    cfg["max_steps"] = 15
    cfg["num_skeletons"] = 100
    cfg["flat_sample"] = False # TODO: check; may cause progressive widening
    cfg['save_dir'] = os.getcwd()+"/runs/run{}".format(time.time())

    # cfg['from_scratch'] = False # imp: re-use!!! but graph gets too big
    cfg['envelope_threshold'] = 0.05 # enforce reuse

    # state
    s = EnvState(holding={ROBOTS[0]:[],ROBOTS[1]:[]},clean=[REGIONS[1]],
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
        
        
        action, info, store = planner.get_action(b, store) 
        

        if action.name == "no-op": # symk planning failure returns no-op; else we would get "nothing_ego"
            bp = copy.deepcopy(b)
            observation = None
            
            # replace previous action with nothing 
            for ac in s.next_actions:
                name,args = ac.split("-")
                args=args.split("%")
                if args[0] == ROBOTS[1]:
                    s.next_actions.remove(ac)
                    s.next_actions.append("nothing_action-"+args[0])
            
            continue # skip the rest of the loop (asking for next action) and repeat MDP solving step
            
        else:
            
            if "a" in planner.print_options:
                logging.info("Action: " + str(action))
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
        # update the state as modified by ego!
        s = copy.deepcopy(env.state)

        # remove previous action (nothing)
        for ac in s.next_actions:
            name,args = ac.split("-")
            args=args.split("%")
            if args[0] == ROBOTS[1]:
                s.next_actions.remove(ac)
                
        
        # get current action
                    
        next_actions = get_next_actions_execute(action,b,store)
        for ac in next_actions:
            s.next_actions.append(ac)
            
    
        # true outcome evaluated in functions
        
    # history.add(env.state, bp, a_bp, None, None, reward, info, store, time.time() - st)
        
    logging.info("=" * 20)

    env.wrapup()

    if not planner.config["real_execute"]:
        save_run_data(history, planner.config["save_dir"])
                
                
                



if __name__ == "__main__":
    # run the main function
    main()
    