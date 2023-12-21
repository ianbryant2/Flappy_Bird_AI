import pygame
from .flappy_mechanics import FlappyGame

class FlappyGameManager:   
    def __init__(self, game : FlappyGame) -> None:
        self.movement = None
        self.score_check = 0
        self.game = game
        self.key_event_up = pygame.event.Event(pygame.KEYDOWN, {'key': self.game._get_inputs()[0]})
        self.upper_dist_offset = 10  #How far away from the pipe the bird has to be in order for certain reward 
        self.lower_dist_offset = 20  
    
    def action_sequence(self, action : list[int]) -> tuple[float, bool, int]:
        '''Will perform the action and return information resulting from the action in a tuple.
        It will be ordered as the reward, if the game is done, and the score of the game'''
        self._action(action, self.game.score)
        #TODO add an accesor to the score
        return self._get_reward(self.game.level_loop(), self.game.score)
    
    def reset(self) -> None:
        '''Will reset the game when it is over'''
        self.game.init_level()
        pygame.event.post(self.key_event_up)

    def get_state(self) -> list[int]:
        '''Will return the state of the game in the shape of a tuple .
        It would be ordered as the distance to the next set of pipes, the distance to the top of the pipe, the distance to the bottom of the pipe, and the birds y velocity'''
        try:
            distance_top = self.game.player_y - (self.game.IMAGES['pipe'][0].get_height() + self.game.upper_pipes[-2]['y'])
            distance_bottom = self.game.lower_pipes[-2]['y'] - self.game.player_y
            return (
                self.game.lower_pipes[-2]['x'],
                distance_top,
                distance_bottom,
                self.game.player_vel_y,
            )
        except AttributeError: #when getting the length during setup of model of the get_state when some variables have not been made yet
            return (0,0,0,0)
    
    def play(self) -> None:
        '''Will play the main loop of the game'''
        while True:
            self.game.init_level()
            crashInfo = None
            while crashInfo == None:
                crashInfo = self.game.level_loop()
            self.game.show_game_over_screen(crashInfo)
    
    #TODO update so that no random attributes are assigned and that the predict does not need to be a tensor 
    def set_outputs(self, predict : list[float]):
        '''Will update values that should be displayed during evaluation'''
        self.game.output1 = predict[0]
        self.game.output2 = predict[1]

    #TODO potentially update action to be a named tuple
    def _action(self, action : list[int], score : int):
        '''Will perform the action that is passed in'''
        self.movement = action
        self.score_check = score
        if self.movement == [1,0]:
            pygame.event.post(self.key_event_up)
            #TODO update so it just displays the scores and do need need seperate functions assign_action and assign_wait
            self.game.assign_action()
        else:
            self.game.assign_wait()

    def _determine_pos_reward(self):
        '''Will determine a reward when the bird is inside of the pipes and will increase the reward as it gets closer between the pipes'''
        return 8.88889*(self.upper_dist_offset) + 355.556

    def _get_reward(self, crash_info : dict[str], score : int) -> tuple[float, bool, int]:
        '''Will get the reward after an action at the current state'''
        reward = 0
        done = False
        distance_top = self.game.player_y - (self.game.IMAGES['pipe'][0].get_height()+self.game.upper_pipes[-2]['y'])
        distance_bottom = self.game.lower_pipes[-2]['y'] - self.game.player_y
        if distance_top < self.upper_dist_offset  or distance_bottom < self.lower_dist_offset :
            reward += -200

        else:
            reward += self._determine_pos_reward()

        if crash_info != None: # Did crash
            done = True
            reward += -200
            return reward, done, score
        
        elif score > self.score_check:
            reward += 200
            return reward, done, score
        
        else:
            return reward, done, score