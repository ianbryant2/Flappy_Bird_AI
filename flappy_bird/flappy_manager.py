import pygame
from .flappy_mechanics import FlappyGame
from collections.abc import Callable

class FlappyGameManager:   
    def __init__(self, game : FlappyGame, game_type : str = None) -> None:
        self._movement = None
        self._score_check = 0
        self._game = game
        self._upper_dist_offset = 10  #How far away from the pipe the bird has to be in order for certain reward 
        self._lower_dist_offset = 20  
        self._game_type = game_type
        #TODO Add so that it checks a view attribute named ai_input that is bool to see if they want ai_input
        #TODO Add an attributes that represents if they want information about AI displayed only if the FlappyGame contains another attribute (this will make it so that play game cannot have)
        #TODO If they want info displayed then add that information to the view not the game, the game should not handle displaying extra information
        #TODO most of above can be in game_delagate but what outputs from the model do what [1...] is up to person making game
        #TODO action should be a list of ints where the 1 in a input means do
        self.input_post = self._game.VIEW.post_input
        self.FLAP_EVENT = [1, 0]
    
    def action_sequence(self, action : list[int]) -> tuple[float, bool, int]:
        '''Will perform the action and return information resulting from the action in a tuple.
        It will be ordered as the reward, if the game is done, and the score of the game'''
        self._action(action, self._game.score)
        #TODO add an accesor to the score
        return self._get_reward(self._game.level_loop(), self._game.score)
    
    def reset(self) -> None:
        '''Will reset the game when it is over'''
        self._game.init_level()
        self.input_post(self.FLAP_EVENT)

    def get_state(self) -> list[int]:
        '''Will return the state of the game in the shape of a tuple .
        It would be ordered as the distance to the next set of pipes, the distance to the top of the pipe, the distance to the bottom of the pipe, and the birds y velocity'''
        try:
            distance_top = self._game.player_y - (self._game.IMAGES['pipe'][0].get_height() + self._game.upper_pipes[-2]['y'])
            distance_bottom = self._game.lower_pipes[-2]['y'] - self._game.player_y
            return (
                self._game.lower_pipes[-2]['x'],
                distance_top,
                distance_bottom,
                self._game.player_vel_y,
            )
        except AttributeError: #when getting the length during setup of model of the get_state when some variables have not been made yet
            return (0,0,0,0)
    
    def play(self) -> None:
        '''Will play the main loop of the game'''
        while True:
            self._game.init_level()
            crashInfo = None
            while crashInfo == None:
                crashInfo = self._game.level_loop()
            self._game.show_game_over_screen(crashInfo)
    
    #TODO update so that no random attributes are assigned
    def set_outputs(self, predict : list[float]):
        '''Will update values that should be displayed during evaluation'''
        self._game.output1 = predict[0]
        self._game.output2 = predict[1]

    #TODO potentially update action to be a named tuple
    def _action(self, action : list[int], score : int):
        '''Will perform the action that is passed in'''
        self._movement = action
        self._score_check = score
        self.input_post(self._movement)
        if self._movement == [1,0]:
            #TODO update so it just displays the scores and do need need seperate functions assign_action and assign_wait
            self._game.assign_action()
        else:
            self._game.assign_wait()

    def _determine_pos_reward(self):
        '''Will determine a reward when the bird is inside of the pipes and will increase the reward as it gets closer between the pipes'''
        return 8.88889 * (self._upper_dist_offset) + 355.556

    def _get_reward(self, crash_info : dict[str], score : int) -> tuple[float, bool, int]:
        '''Will get the reward after an action at the current state and return it and information about if the game is done and what the current score is'''
        reward = 0
        done = False
        distance_top = self._game.player_y - (self._game.IMAGES['pipe'][0].get_height() + self._game.upper_pipes[-2]['y'])
        distance_bottom = self._game.lower_pipes[-2]['y'] - self._game.player_y
        if distance_top < self._upper_dist_offset  or distance_bottom < self._lower_dist_offset :
            reward += -200

        else:
            reward += self._determine_pos_reward()

        if crash_info != None: # Did crash
            done = True
            reward += -200
            return reward, done, score
        
        elif score > self._score_check:
            reward += 200
            return reward, done, score
        
        else:
            return reward, done, score
        