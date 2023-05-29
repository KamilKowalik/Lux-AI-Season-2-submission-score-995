from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import pandas as pd
import sys
from scipy.ndimage import distance_transform_cdt

def manhattan_distance(binary_mask):
    distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
    return distance_map

# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_arrays = {1: np.array([0, -1]), 2: np.array([1, 0]), 3: np.array([0, 1]), 4: np.array([-1, 0])}
def direction_to_mod_2(src, target, opp_factories_map, step, step_thr = 20):
    move_signs_hor = {1: 2, -1: 4}
    move_signs_vert = {1: 3, -1: 1}

    move_dict_vert = {1:-1, 3:1}
    move_dict_hor = {2:1, 4:-1}
    ds = target - src
    dx = ds[0]
    dy = ds[1]
 
    if dx == 0 and dy == 0:
        move_candidate = 0
        pos_after = src
    if dx > 0:
        move_candidate_x = 2
        pos_after_x = src + np.array([1,0])
    elif dx < 0:
        move_candidate_x = 4
        pos_after_x= src + np.array([-1,0])
    else:
        move_candidate_x = 0
        pos_after_x = src 
    if dy > 0:
        move_candidate_y = 3
        pos_after_y = src + np.array([0,1])
    elif dy < 0:
        move_candidate_y = 1
        pos_after_y = src + np.array([0,-1])
    else:
        move_candidate_y = 0
        pos_after_y = src
    
    move_candidates = []
    pos_after_x = (pos_after_x[0], pos_after_x[1])
    pos_after_y = (pos_after_y[0], pos_after_y[1])
    rejected_moves=-1
    if pos_after_x not in opp_factories_map:
        move_candidates.append(move_candidate_x)
    else:
        rejected_moves=move_candidate_x
    if pos_after_y not in opp_factories_map:
        move_candidates.append(move_candidate_y)
    else:
        rejected_moves=move_candidate_y
    
    if dx == 0 and dy == 0:
        move = 0
    else:
        move_candidates = [mc for mc in move_candidates if mc != 0]
        if len(move_candidates)>0:
            move = np.random.choice(move_candidates)
        else:
            if rejected_moves in [1,3]:
                for i in [1,-1,2,-2,3,-3]:
                    pos_after_move = src + np.array([i, move_dict_vert[rejected_moves]])
                    pos_after_move = (pos_after_move[0], pos_after_move[1])
                    if (pos_after_move not in opp_factories_map) and pos_after_move[0]>=0 and pos_after_move[1]>=0:
                        move_dir = np.sign(i)
                        move_times = abs(i)
                        move = [move_signs_hor[move_dir]]*move_times + [rejected_moves]
                        break
            elif rejected_moves in [2,4]:
                for i in [1,-1,2,-2,3,-3]:
                    pos_after_move = src + np.array([move_dict_hor[rejected_moves],i])
                    pos_after_move = (pos_after_move[0], pos_after_move[1])
                    if (pos_after_move not in opp_factories_map) and 0<=pos_after_move[0]<=47 and 0<=pos_after_move[1]<=47:
                        move_dir = np.sign(i)
                        move_times = abs(i)
                        move = [move_signs_vert[move_dir]]*move_times + [rejected_moves]
                        break
    return move

def simple_manh_distance(src, target):
    ds=src-target
    return abs(ds[0])+abs(ds[1])

def square3x3(pos):
    f_x, f_y = pos[0], pos[1]
    map3x3 = []
    for x in range(f_x-1,f_x+2):
        for y in range(f_y-1,f_y+2):
            map3x3.append(np.array([x,y]))
    return map3x3

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
            
        self.my_robots_move_queue = {}
        self.my_robots_parent_factory = {}
        self.hunting_dict = {}
        self.opp_robots = {}
        
        self.water_start_turn = 300 #no of turns before game finishes when we start watering
        self.battery_charge_start = 0.1 #percentage of battery when we start charging our heavy robots
        self.ice_return_value = 200 #amount of ice triggering return to the factory
        self.max_hunting_distance = 15 #max distance to find other heavy robots to destroy
        self.min_power_to_start_hunting = 1900 #min power of the hunter robot to start hunting
    
    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
#                 potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
#                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
#                 return dict(spawn=spawn_loc, metal=150, water=150)
            
                valid_spawn_locations = obs["board"]["valid_spawns_mask"]
                ice = obs["board"]["ice"]
                dist_ice = manhattan_distance(1-ice)
                start_map = valid_spawn_locations*(dist_ice.max()-dist_ice)
                start_loc_candidates = np.argwhere(start_map==start_map.max())
#                 spawn_loc = start_loc_candidates[np.random.randint(0, len(start_loc_candidates))]
                
                rubble = obs["board"]["rubble"]
                spawn_candidate_scores = {}
                for i, spawn_loc_candidate in enumerate(start_loc_candidates):
                    sp_x, sp_y = spawn_loc_candidate
                    rub_range = 2
                    close_rubble_map = rubble[sp_x-rub_range:sp_x+rub_range+1, sp_y-rub_range:sp_y+rub_range+1]
                    score = (close_rubble_map==0).mean()
                    spawn_candidate_scores[i]=score

                key_of_best_candidate = max(spawn_candidate_scores, key=spawn_candidate_scores.get)
                spawn_loc =start_loc_candidates[key_of_best_candidate]
                return dict(spawn=spawn_loc, metal=150, water=150)

            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        #print(self.player, step)
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        opp_factories = game_state.factories[self.opp_player]
        opp_factories_map = []
        for unit_id, factory in opp_factories.items():
            f_x, f_y = factory.pos
            for x in range(f_x-1,f_x+2):
                for y in range(f_y-1,f_y+2):
                    opp_factories_map.append((x,y))
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        #
        for unit_id, unit in game_state.units[self.opp_player].items():
            if unit_id not in self.opp_robots.keys():
                self.opp_robots[unit_id] = {'pos': 0, 'sit_counter':0, 'target': False, 'alive': True, 'type': ''}  
            prev_pos = self.opp_robots[unit_id]['pos']
            cur_pos = unit.pos
            if (prev_pos==cur_pos).mean()==1:
                self.opp_robots[unit_id]['sit_counter']+=1
            else:
                self.opp_robots[unit_id]['sit_counter']=0
            
            if self.opp_robots[unit_id]['sit_counter'] >= 3:
                self.opp_robots[unit_id]['target'] = True
            self.opp_robots[unit_id]['pos'] = cur_pos
            self.opp_robots[unit_id]['type'] = unit.unit_type

        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if (self.env_cfg.max_episode_length - game_state.real_env_steps < self.water_start_turn) and (factory.cargo.water > 50):
                if factory.water_cost(game_state) <= factory.cargo.water:
                    actions[unit_id] = factory.water()
            factory_tiles+=square3x3(factory.pos) 
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        for u_id, u in units.items():
            if u_id not in self.my_robots_parent_factory.keys():
                self.my_robots_parent_factory[u_id] = square3x3(u.pos)

        regular_units = {key:value for (key,value) in units.items() if key not in self.hunting_dict.keys()}
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in regular_units.items():
            #basic stats
            battery_capacity = 150 if unit.unit_type == "LIGHT" else 3000
            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_tiles = self.my_robots_parent_factory[unit_id]
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                #closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # previous ice mining code
                if (unit.power < battery_capacity * self.battery_charge_start) & adjacent_to_factory:
                    actions[unit_id] = [unit.pickup(4, battery_capacity-unit.power)]
                elif unit.cargo.ice < self.ice_return_value:
                    ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=1, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= self.ice_return_value:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0, n=1)]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        #targets
        if True:
#         if self.player == 'player_1':
            clean_list_h = []
            clean_list_t = []            
            for h,t in self.hunting_dict.items():
                if t not in game_state.units[self.opp_player].keys():
                    clean_list_h.append(h)
                    clean_list_t.append(t)
            for t in clean_list_t:
                try:
                    self.opp_robots.pop(t)
                except:
                    print('DEBUG - unhandled error')
            for h in clean_list_h:
                self.hunting_dict.pop(h)
            def simple_manh_distance(src, target):
                ds=src-target
                return abs(ds[0])+abs(ds[1])
            
            hunting_pair_candidates = []
            for opp_unit_id, opp_unit in self.opp_robots.items():
                min_dist_to_target = 10000
                closest_unit_id = ''
                for unit_id, unit in units.items():
                    if (opp_unit['target'] == True) and (opp_unit['type'] == 'HEAVY') and (unit_id not in self.hunting_dict.keys()) and (opp_unit_id not in self.hunting_dict.values()):
                        distance = simple_manh_distance(unit.pos, opp_unit['pos'])
                        if distance < min_dist_to_target:
                            min_dist_to_target = distance
                            closest_unit_id = unit_id
                if min_dist_to_target <= self.max_hunting_distance and (units[closest_unit_id].power > self.min_power_to_start_hunting) :
                    hunting_pair_candidates.append({'opp_unit_id':opp_unit_id,'unit_id': closest_unit_id, 'distance': min_dist_to_target})
            
            if len(hunting_pair_candidates)>0:
                hunting_pair_candidates = pd.DataFrame(hunting_pair_candidates).sort_values('distance').drop_duplicates('unit_id', keep='first')
                for i, row in hunting_pair_candidates.iterrows():
                    self.hunting_dict[row['unit_id']]= row['opp_unit_id']
                    break
                
            dead_hunters=[]
            for hunter,target in self.hunting_dict.items():
                if hunter not in units.keys():
                    dead_hunters.append(hunter)

            for dh in dead_hunters:
                try:
                    self.hunting_dict.pop(dh)
                except:
                    pass
            for hunter,target in self.hunting_dict.items():        
                hunter_unit = units[hunter]
                direction = direction_to_mod_2(hunter_unit.pos, self.opp_robots[target]['pos'], opp_factories_map, step, 1000)
                if (type(direction) == list) and hunter not in self.my_robots_move_queue.keys():
                    self.my_robots_move_queue[hunter]=direction
                if hunter in self.my_robots_move_queue.keys():
                    direction = self.my_robots_move_queue[hunter][0]
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and hunter_unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[hunter] = [hunter_unit.move(direction, repeat=0, n=1)]
                        self.my_robots_move_queue[hunter]=self.my_robots_move_queue[hunter][1:]
                        if len(self.my_robots_move_queue[hunter])==0:
                            self.my_robots_move_queue.pop(hunter)
                else:
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and hunter_unit.power >= move_cost + unit.action_queue_cost(game_state):
                        actions[hunter] = [hunter_unit.move(direction, repeat=0, n=1)]
                        #print('ah', actions[hunter])
        actions_before_coord = []
        for unit_id, unit in units.items():
            cur_pos = unit.pos
            if unit_id in actions.keys():
                cur_action = actions[unit_id]
                action_type = cur_action[0][0]
                if action_type == 0:
                    action_dir = cur_action[0][1]
                    pos_after = cur_pos + move_arrays[action_dir]                    
                else:
                    pos_after = cur_pos
            else:
                action_type = 99
                pos_after = cur_pos
            actions_before_coord.append({'unit_id':unit_id, 'action_type': action_type, 'pos_after':str(pos_after)})
        if len(actions_before_coord)>0:
            actions_before_coord = pd.DataFrame(actions_before_coord)
            actions_before_coord_wo_dupl = actions_before_coord.sort_values('action_type').drop_duplicates('pos_after', keep = 'last') 
            robots_to_wait = set(actions_before_coord['unit_id'].unique()).difference(actions_before_coord_wo_dupl['unit_id'].unique())
            for unit_id in robots_to_wait:
                actions.pop(unit_id)
        return actions
