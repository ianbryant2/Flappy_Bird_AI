#Will need things for handling input, displaying, and making audio 
# Should also handle fps

# def draw(drawing_list list[tuples[surface : tuple[cord]]], function_to_run_before_drawing : callable)

import pygame
import sys
from collections.abc import Callable

from pygame.mixer import Sound

class BaseView():
    '''Will be the baseclass that can be inherited from in order to implement the specific view functions'''
    def __init__(self, **kwargs) -> None:
        pygame.display.set_mode((1,1))
        pygame.init()

    def draw_display(self, display_elements : list[tuple[pygame.Surface, tuple[int, int]]], func : Callable = None) -> None:
        '''When implemented, will display the elements and run any setup function before drawing'''
        pass

    def play_audio(self, audio : pygame.mixer.Sound) -> None:
        '''When implemented, will play the audi that was passed in'''
        pass

    def handle_input(self, flap_inputs : list[int], func : Callable) -> 'Return_type_of_func':
        '''Will handle the inputs of the game and run the function func if there is an input in the defined flap_inputs.
        The return type is the return type of the function'''
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key in flap_inputs):
                return func()

class PlayView(BaseView):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.FPS_CLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((kwargs['width'], kwargs['height']))
        pygame.display.set_caption('Flappy Bird')
        self.FPS = kwargs['fps']
        

    def draw_display(self, display_elements : list[tuple[pygame.Surface, tuple[int, int]]], func : Callable = None) -> None:
        '''Will display the elements in a pygame window and run any setup function before drawing
        The elements should be ordered where the first element will be drawn first and the last element will be drawn last'''
        for element in display_elements:
            self.SCREEN.blit(element[0], element[1])
        pygame.display.update()
        self.FPS_CLOCK.tick(self.FPS)

    def play_audio(self, audio: pygame.mixer.Sound) -> None:
        '''Will play the passed in sound'''
        audio.play()

class TrainView(BaseView):
    pass

class EvaluateView(PlayView):

    def __init__(self, show_display=True, **kwargs) -> None:
        super().__init__(**kwargs)
        if not show_display:
            self.draw_display = lambda: None

    def play_audio(self, audio: Sound) -> None:
        pass