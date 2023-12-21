import pygame
from .flappy_mechanics import FlappyGame, SPRITES_DIR, AUDIOS_DIR

class FlappyPlayGame(FlappyGame):
    def __init__(self, file_dir = ''):
        FlappyGame.__init__(self, file_dir = file_dir)

class FlappyTrainGame(FlappyGame):
    def __init__(self, file_dir = ''):
        FlappyGame.__init__(self, file_dir)

    def _get_fps(self):
        return 3840

    def _get_inputs(self):
        return [pygame.K_AC_BACK] #Using Andriod Backspace Key because not on PC keyboard

    def _intro_looper(self):
        '''First game loop in intro screen'''
        return {
        'playery': self.player_y + self.player_shm_vals['val'],
        'basex': self.base_x,
        'playerIndexGen': self.player_index_gen,
                }
    def _wing_sound(self):
        pass
    
    def _point_sound(self):
        pass
    
    def _hit_sound(self):
        pass
    
    def _die_sound(self):
        pass

class FlappyEvaluateGame(FlappyGame):
    def __init__(self, file_dir = '', fps = 3840):
        FlappyGame.__init__(self, file_dir)
        self.fps = fps
        self.font = pygame.font.SysFont('Courier New', 30) #Name of font then size
        self.IMAGES['wait'] = pygame.image.load(file_dir + SPRITES_DIR + '/added/Wait.png').convert_alpha()
        self.IMAGES['flap'] = pygame.image.load(file_dir + SPRITES_DIR + '/added/Flap.png').convert_alpha()
        self.output1 = None
        self.output2 = None
        self.flapCount = 0 #Used to see how many frames have passed since the flap is shown
        self.imageShown = None

    def _get_fps(self) -> int:
        return self.fps

    def _get_inputs(self):
        return [pygame.K_AC_BACK] #Using Andriod Backspace Key because not on PC keyboard
        
    def _intro_looper(self):
        '''First game loop in intro screen'''
        return {
        'playery': self.player_y + self.player_shm_vals['val'],
        'basex': self.base_x,
        'playerIndexGen': self.player_index_gen,
            }

    def assign_action(self):
        self.imageShown = self.IMAGES['flap']
        self.flapCount = 1
    
    def assign_wait(self):
        if self.flapCount > 2:
            self.flapCount = 0
            self.imageShown = self.IMAGES['wait']
        else:
            self.flapCount += 1

    def _show_info(self):
        if self.imageShown != None:
            self.SCREEN.blit(self.imageShown, (150,425))
            tOutput1 = self.font.render(str(int(self.output1)), True, (0,0,0))
            tOutput2 = self.font.render(str(int(self.output2)), True, (0,0,0))
            self.SCREEN.blit(tOutput1, (10,435))
            self.SCREEN.blit(tOutput2, (10,475))

    def _wing_sound(self):
        pass
    
    def _point_sound(self):
        pass
    
    def _hit_sound(self):
        pass
    
    def _die_sound(self):
        pass