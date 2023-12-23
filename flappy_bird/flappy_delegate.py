import pygame
import sys
from collections.abc import Callable

class BaseView():
    '''Will be the baseclass that can be inherited from in order to implement the specific view methods'''
    def __init__(self, **kwargs) -> None:
        pygame.display.set_mode((1,1))
        pygame.init()
        self.FPS_CLOCK = pygame.time.Clock()
        self.flap_input = False

    def draw_display(self, display_elements : list[tuple[pygame.Surface, tuple[int, int]]], func : Callable = None) -> None:
        '''When implemented, will display the elements and run any setup function before drawing'''
        pass

    def play_audio(self, audio : pygame.mixer.Sound) -> None:
        '''When implemented, will play the audi that was passed in'''
        pass

    def handle_input(self, flap_inputs : list[int], func : Callable) -> 'None | Return_type_of_func':
        '''Will handle the inputs of the game and run the function func if there is an input in the defined flap_inputs.
        The return type is the return type of the function or None if it is not called'''
        pass
    
    def _pygame_input(self, flap_inputs : list[int], func : Callable) -> 'None | Return_type_of_func':
        '''Handles input when flap input utilizes pygame'''
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key in flap_inputs:
                return func()
            
    def _ai_input(self, flap_inputs : list[int], func : Callable) -> 'None | Return_type_of_func':
        '''Handles input when flap input does not utilize pygame'''
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        if self.flap_input:
            self.flap_input = False
            return func()
    
    def _init_display(self, width : int, height : int, fps : int) -> None:
        '''Initilizes pygame and its display'''
        self.SCREEN = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Flappy Bird')
        self.FPS = fps

    def _pygame_display(self, display_elements : list[tuple[pygame.Surface, tuple[int, int]]], func : Callable = None) -> None:
        '''NEED TO RUN DISPLAY INIT OR WILL CAUSE ERROR
        Handles displaying when wanting to display in pygame window
        Will display the elements in a pygame window and run any setup function before drawing
        The elements should be ordered where the first element will be drawn first and the last element will be drawn last
        Any setup before displaying should be passed in as func'''
        if func != None:
            func()
        for element in display_elements:
            self.SCREEN.blit(element[0], element[1])
        pygame.display.update()
        self.FPS_CLOCK.tick(self.FPS)


class PlayView(BaseView):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_display(kwargs['width'], kwargs['height'], kwargs['fps'])
        
    def draw_display(self, display_elements : list[tuple[pygame.Surface, tuple[int, int]]], func : Callable = None) -> None:
        self._pygame_display(display_elements, func)

    def play_audio(self, audio: pygame.mixer.Sound) -> None:
        '''Will play the passed in sound'''
        audio.play()

class EvaluateView(BaseView):

    def __init__(self, show_display=True, **kwargs) -> None:
        super().__init__(**kwargs)
        if show_display:
            self._init_display(kwargs['width'], kwargs['height'], kwargs['fps'])
            self.draw_display = self._pygame_display

    def handle_input(self, flap_inputs: list[int], func: Callable) -> None:
        return self._ai_input(flap_inputs,  func)

class TrainView(BaseView):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def handle_input(self, flap_inputs: list[int], func: Callable) -> 'None | Return_type_of_func':
        return self._ai_input(flap_inputs, func)