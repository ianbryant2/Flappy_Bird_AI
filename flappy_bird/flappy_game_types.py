import pygame
from flappy_bird.flappy_mechanics import FlappyGame, SPRITES_DIR, AUDIOS_DIR, SCREEN_HEIGHT, SCREEN_WIDTH
from flappy_bird.flappy_delegate import PlayView, TrainView, EvaluateView

class FlappyPlayGame(FlappyGame):
    def __init__(self, file_dir = ''):
        view = PlayView(fps=30, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        FlappyGame.__init__(self, view, file_dir = file_dir)

class FlappyTrainGame(FlappyGame):
    def __init__(self, file_dir=''):
        view = TrainView(fps=30000, width=SCREEN_WIDTH, height=SCREEN_HEIGHT) #pass in kwargs if we wawnt to change to display
        FlappyGame.__init__(self, view, file_dir)

    def _get_inputs(self):
        return [pygame.K_AC_BACK] #Using Andriod Backspace Key because not on PC keyboard

    def _intro_looper(self):
        '''First game loop in intro screen'''
        return {
        'playery': self.player_y + self.player_shm_vals['val'],
        'basex': self.base_x,
        'playerIndexGen': self.player_index_gen,
                }

class FlappyEvaluateGame(FlappyGame):
    def __init__(self, file_dir = ''):
        view = EvaluateView(fps=30, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        FlappyGame.__init__(self, view, file_dir)
        self.font = pygame.font.SysFont('Courier New', 30) #Name of font then size
        self.IMAGES['wait'] = pygame.image.load(file_dir + SPRITES_DIR + '/added/Wait.png').convert_alpha()
        self.IMAGES['flap'] = pygame.image.load(file_dir + SPRITES_DIR + '/added/Flap.png').convert_alpha()
        self.output1 = None
        self.output2 = None
        self.flap_count = 0 #Used to see how many frames have passed since the flap is shown
        self.image_shown = None

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
        self.image_shown = self.IMAGES['flap']
        self.flap_count = 1
    
    def assign_wait(self):
        if self.flap_count > 2:
            self.flap_count = 0
            self.image_shown = self.IMAGES['wait']
        else:
            self.flap_count += 1

    def _show_info(self) -> list[tuple[pygame.Surface], tuple[int, int]]:
        list_images = []

        if self.image_shown != None:
            list_images.append((self.image_shown, (150,425)))
            tOutput1 = self.font.render(str(int(self.output1)), True, (0,0,0))
            tOutput2 = self.font.render(str(int(self.output2)), True, (0,0,0))
            list_images.append((tOutput1, (10,435)))
            list_images.append((tOutput2, (10,475)))
        return list_images
