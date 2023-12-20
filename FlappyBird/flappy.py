from itertools import cycle
import random
import sys
import pygame

class Game:
    def __init__(self, file_dir : str = '') -> None:
        self.SCREEN_WIDTH  = 288
        self.SCREEN_HEIGHT = 512
        self.PIPE_GAP_SIZE  = 100 # gap between upper and lower part of pipe
        self.BASE_Y        = self.SCREEN_HEIGHT * 0.79

        self.IMAGES : dict[str, pygame.Surface | tuple[pygame.Surface]]
        self.SOUNDS : dict[str, pygame.mixer.Sound]
        self.HITMASKS : dict[str, tuple[list[list[bool]]]]

        self.IMAGES, self.SOUNDS, self.HITMASKS = dict(), dict(), dict()

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = (
            # red bird
            (
                file_dir + '/FlappyBird/assets/sprites/redbird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/redbird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/redbird-downflap.png',
            ),
            # blue bird
            (
                file_dir + '/FlappyBird/assets/sprites/bluebird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/bluebird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/bluebird-downflap.png',
            ),
            # yellow bird
            (
                file_dir + '/FlappyBird/assets/sprites/yellowbird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/yellowbird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/yellowbird-downflap.png',
            ),
        )

        # list of backgrounds
        self.BACKGROUNDS_LIST = (
            file_dir + '/FlappyBird/assets/sprites/background-day.png',
            file_dir + '/FlappyBird/assets/sprites/background-night.png',
        )

        # list of pipes
        self.PIPES_LIST = (
            file_dir + '/FlappyBird/assets/sprites/pipe-green.png',
            file_dir + '/FlappyBird/assets/sprites/pipe-red.png',
        )


        
        self.xrange = range


        pygame.init()
        self.FPS_CLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/0.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/1.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/2.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/3.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/4.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/5.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/6.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/7.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/8.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            self.sound_ext = '.wav'
        else:
            self.sound_ext = '.ogg'

        self.SOUNDS['die']    = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/die' + self.sound_ext)
        self.SOUNDS['hit']    = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/hit' + self.sound_ext)
        self.SOUNDS['point']  = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/point' + self.sound_ext)
        self.SOUNDS['swoosh'] = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/swoosh' + self.sound_ext)
        self.SOUNDS['wing']   = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/wing' + self.sound_ext)



        
    def init_level(self) -> None:
        '''Used to initilize the level'''
        # select random background sprites
        self.rand_bg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)

        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[self.rand_bg]).convert()
        # select random player sprites
        self.rand_player = random.randint(0, len(self.PLAYERS_LIST) - 1)

        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[self.rand_player][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[self.rand_player][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[self.rand_player][2]).convert_alpha(),
        )

        # select random pipe sprites
        self.pipe_index = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(self.PIPES_LIST[self.pipe_index]).convert_alpha(), False, True),
            pygame.image.load(self.PIPES_LIST[self.pipe_index]).convert_alpha(),
        )

        # hitmask for pipes
        self.HITMASKS['pipe'] = (
            self._get_hitmask(self.IMAGES['pipe'][0]),
            self._get_hitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self._get_hitmask(self.IMAGES['player'][0]),
            self._get_hitmask(self.IMAGES['player'][1]),
            self._get_hitmask(self.IMAGES['player'][2]),
        )
        
        # index of player to blit on screen
        self.player_index = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        self.loop_iter = 0

        self.player_x = int(self.SCREEN_WIDTH * 0.2)
        self.player_y = int((self.SCREEN_HEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        self.message_x = int((self.SCREEN_WIDTH - self.IMAGES['message'].get_width()) / 2)
        self.message_y = int(self.SCREEN_HEIGHT * 0.12)

        self.base_x = 0
        # amount by which base can maximum shift to left
        self.base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        self.player_shm_vals = {'val': 0, 'dir': 1}
        self.movement_info = self._show_welcome_animation()

        self.score = self.player_index = self.loop_iter = 0
        self.player_index_gen = self.movement_info['playerIndexGen']
        self.player_x, self.player_y = int(self.SCREEN_WIDTH * 0.2), self.movement_info['playery']
        self.base_x = self.movement_info['basex']
        self.base_shift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.new_pipe1 = self._get_random_pipe()
        self.new_pipe2 = self._get_random_pipe()

        self.upper_pipes : list[dict[str, int]]
        self.lower_pipes : list[dict[str, int]]

        self.upper_pipes = [
            {'x': self.SCREEN_WIDTH + 200, 'y': self.new_pipe1[0]['y']},
            {'x': self.SCREEN_WIDTH + 200 + (self.SCREEN_WIDTH / 2), 'y': self.new_pipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lower_pipes = [
            {'x': self.SCREEN_WIDTH + 200, 'y': self.new_pipe1[1]['y']},
            {'x': self.SCREEN_WIDTH + 200 + (self.SCREEN_WIDTH / 2), 'y': self.new_pipe2[1]['y']},
        ]

        self.pipe_vel_x = -4 

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.player_vel_y    =  -9   # player's velocity along Y, default same as playerFlapped
        self.player_max_vel_y =  10   # max vel along Y, max descend speed
        self.player_min_vel_y =  -8   # min vel along Y, max ascend speed
        self.player_acc_y    =   1   # players downward acceleration
        self.player_rot     =  45   # player's rotation
        self.player_vel_rot  =   3   # angular speed
        self.player_rot_thr  =  20   # rotation threshold
        self.player_flap_acc =  -9   # players speed on flapping
        self.player_flapped = False # True when player flaps

    def level_loop(self) -> dict[str]:
        '''completes one game loop'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key in self._get_inputs()):
                if self.player_y > -2 * self.IMAGES['player'][0].get_height():
                    self.player_vel_y = self.player_flap_acc
                    self.player_flapped = True
                    self._wing_sound()
        
        # check for crash here
        self.crash_test = self._check_crash({'x': self.player_x, 'y': self.player_y, 'index': self.player_index},
                               self.upper_pipes, self.lower_pipes)
        if self.crash_test[0]:
            return {
                'y': self.player_y,
                'groundCrash': self.crash_test[1],
                'basex': self.base_x,
                'upperPipes': self.upper_pipes,
                'lowerPipes': self.lower_pipes,
                'score': self.score,
                'playerVelY': self.player_vel_y,
                'playerRot': self.player_rot
            }

        # check for score
        self.player_mid_pos = self.player_x + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upper_pipes:
            self.pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if self.pipeMidPos <= self.player_mid_pos < self.pipeMidPos + 4:
                self.score += 1
                self._point_sound()

        # playerIndex basex change
        if (self.loop_iter + 1) % 3 == 0:
            self.player_index = next(self.player_index_gen)
        self.loop_iter = (self.loop_iter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # rotate the player
        if self.player_rot > -90:
            self.player_rot -= self.player_vel_rot

        # player's movement
        if self.player_vel_y < self.player_max_vel_y and not self.player_flapped:
            self.player_vel_y += self.player_acc_y
        if self.player_flapped:
            self.player_flapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.player_rot = 45

        self.player_height = self.IMAGES['player'][self.player_index].get_height()
        self.player_y += min(self.player_vel_y, self.BASE_Y - self.player_y - self.player_height)

        # move pipes to left
        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            u_pipe['x'] += self.pipe_vel_x
            l_pipe['x'] += self.pipe_vel_x

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]['x'] < 5:  #at faster frame rate sometimes wont generate pipe
            self.newPipe = self._get_random_pipe()
            self.upper_pipes.append(self.newPipe[0])
            self.lower_pipes.append(self.newPipe[1])

        # remove first pipe if its out of the screen
        if len(self.upper_pipes) > 0 and self.upper_pipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for u_pipe, l_pipe in zip(self.upper_pipes, self.lower_pipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (u_pipe['x'], u_pipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (l_pipe['x'], l_pipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.base_x, self.BASE_Y))
        # print score so player overlaps the score
        self._show_score(self.score)

        # Player rotation has a threshold
        self.visible_rot = self.player_rot_thr
        if self.player_rot <= self.player_rot_thr:
            self.visible_rot = self.player_rot
        
        self.player_surface = pygame.transform.rotate(self.IMAGES['player'][self.player_index], self.visible_rot)
        self.SCREEN.blit(self.player_surface, (self.player_x, self.player_y))
        self._show_info()
        pygame.display.update()
        self.FPS_CLOCK.tick(self._get_fps())

    def show_game_over_screen(self, crash_info : dict[str]) -> None:
        '''crashes the player down and shows gameover image'''
        score = crash_info['score']
        playerx = self.SCREEN_WIDTH * 0.2
        playery = crash_info['y']
        player_height = self.IMAGES['player'][0].get_height()
        player_velY = crash_info['playerVelY']
        player_accY = 2
        player_rot = crash_info['playerRot']
        player_vel_rot = 7

        self.base_x = crash_info['basex']

        upperPipes, lowerPipes = crash_info['upperPipes'], crash_info['lowerPipes']

        # play hit and die sounds
        self._hit_sound()
        if not crash_info['groundCrash']:
            pass
            self._die_sound()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and (event.key in self._get_inputs()):
                    if playery + player_height >= self.BASE_Y - 1:
                        return

            # player y shift
            if playery + player_height < self.BASE_Y - 1:
                playery += min(player_velY, self.BASE_Y - playery - player_height)

            # player velocity change
            if player_velY < 15:
                player_velY += player_accY

            # rotate only when it's a pipe crash
            if not crash_info['groundCrash']:
                if player_rot > -90:
                    player_rot -= player_vel_rot

            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.SCREEN.blit(self.IMAGES['base'], (self.base_x, self.BASE_Y))
            self._show_score(score)
            player_surface = pygame.transform.rotate(self.IMAGES['player'][1], player_rot)
            self.SCREEN.blit(player_surface, (playerx,playery))
            self.SCREEN.blit(self.IMAGES['gameover'], (50, 180))
            self.FPS_CLOCK.tick(self._get_fps())
            pygame.display.update()

    #TODO update so function does not need to be defined
    def assign_action(self) -> None:
        pass

    def assign_wait(self) -> None:
        pass

    def _show_welcome_animation(self) -> dict[str, int]:
        '''shows the welcome animation'''
        while True:

            values = self._intro_looper()

            # adjust playery, playerIndex, basex
            if (self.loop_iter + 1) % 5 == 0:
                self.player_index = next(self.player_index_gen)
            self.loop_iter = (self.loop_iter + 1) % 30
            self.base_x = -((-self.base_x + 4) % self.base_shift)
            self._player_shm(self.player_shm_vals)

            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))
            self.SCREEN.blit(self.IMAGES['player'][self.player_index],
                        (self.player_x, self.player_y + self.player_shm_vals['val']))
            self.SCREEN.blit(self.IMAGES['message'], (self.message_x, self.message_y))
            self.SCREEN.blit(self.IMAGES['base'], (self.base_x, self.BASE_Y))

            pygame.display.update()
            self.FPS_CLOCK.tick(self._get_fps())

            if values != None:
                return values
    
    def _check_crash(self, player : dict[str, int], upper_pipes : list[dict[str, int]], lower_pipes : list[dict[str, int]]) -> bool:
        '''Returns True if player collides with base or pipes.'''
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASE_Y - 1:
            return [True, True]
        else:

            player_rect = pygame.Rect(player['x'], player['y'],
                        player['w'], player['h'])
            pipe_w = self.IMAGES['pipe'][0].get_width()
            pipe_h = self.IMAGES['pipe'][0].get_height()

            for u_pipe, l_pipe in zip(upper_pipes, lower_pipes):
                # upper and lower pipe rects
                u_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], pipe_w, pipe_h)
                l_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], pipe_w, pipe_h)

                # player and upper/lower pipe hitmasks
                p_hitmask = self.HITMASKS['player'][pi]
                u_hitmask = self.HITMASKS['pipe'][0]
                l_hitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                u_collide = self._pixel_collision(player_rect, u_pipe_rect, p_hitmask, u_hitmask)
                l_collide = self._pixel_collision(player_rect, l_pipe_rect, p_hitmask, l_hitmask)

                if u_collide or l_collide:
                    return [True, False]

        return [False, False]

    def _show_score(self,score) -> None:
        '''displays score in center of screen'''
        score_digits = [int(x) for x in list(str(score))]
        total_width = 0 # total width of all numbers to be printed

        for digit in score_digits:
            total_width += self.IMAGES['numbers'][digit].get_width()

        x_offset = (self.SCREEN_WIDTH - total_width) / 2

        for digit in score_digits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (x_offset, self.SCREEN_HEIGHT * 0.1))
            x_offset += self.IMAGES['numbers'][digit].get_width()

    def _get_random_pipe(self) -> list[dict[str, int]]:
        '''returns a randomly generated pipe'''
        # y of gap between upper and lower pipe
        gap_y = random.randrange(0, int(self.BASE_Y * 0.6 - self.PIPE_GAP_SIZE))
        gap_y += int(self.BASE_Y * 0.2)
        pipe_height = self.IMAGES['pipe'][0].get_height()
        pipe_x = self.SCREEN_WIDTH + 10

        return [
            {'x': pipe_x, 'y': gap_y-pipe_height},  # upper pipe
            {'x': pipe_x, 'y': gap_y+self.PIPE_GAP_SIZE}, # lower pipe
        ]

    def _player_shm(self, player_shm : dict[str, int]) -> None:
        '''oscillates the value of playerShm['val'] between 8 and -8'''
        if abs(player_shm['val']) == 8:
            player_shm['dir'] *= -1

        if player_shm['dir'] == 1:
            player_shm['val'] += 1
        else:
            player_shm['val'] -= 1

    def _pixel_collision(self, rect1 : pygame.Rect, rect2 : pygame.Rect, hitmask1 : list[list[bool]], hitmask2 : list[list[bool]]) -> bool: 
        '''Checks if two objects collide and not just their rects'''
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in self.xrange(rect.width):
            for y in self.xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
        
    def _get_hitmask(self, image: pygame.Surface) -> list[list[bool]]:
        '''returns a hitmask using an image's alpha.'''
        mask : list[list[bool]]
        mask = []
        for x in self.xrange(image.get_width()):
            mask.append([])
            for y in self.xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask
    
    def _show_info(self) -> None:
        '''Would show infor at the bottom of the screen'''
        pass

    def _get_fps(self) -> int:
        '''Returns the FPS for current game'''
        return 30
    
    def _get_inputs(self) -> list[int]:
        '''Returns the correct input for current game'''
        return [pygame.K_SPACE, pygame.K_UP]
    
    def _intro_looper(self) -> dict[str, int]:
        '''First game loop in intro screen'''
        # May need to be differnt for training and evaulating 
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key in self._get_inputs()):
                self._wing_sound()
                return {
                    'playery': self.player_y + self.player_shm_vals['val'],
                    'basex': self.base_x,
                    'playerIndexGen': self.player_index_gen,
                    }
            
    def _wing_sound(self) -> None:
        '''Will play the sound for when the bird flaps'''
        self.SOUNDS['wing'].play()
    
    def _point_sound(self) -> None:
        '''Will play the sound for when a point is achieved'''
        self.SOUNDS['point'].play()
    
    def _hit_sound(self) -> None:
        '''Will play the sound for when the bird hits something'''
        self.SOUNDS['hit'].play()
    
    def _die_sound(self) -> None:
        '''Will play the sound when the bird dies'''
        self.SOUNDS['die'].play()



class PlayGame(Game):
    def __init__(self, file_dir = ''):
        Game.__init__(self, file_dir = file_dir)

class TrainGame(Game):
    def __init__(self, file_dir = ''):
        Game.__init__(self, file_dir)

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

class EvaluateGame(Game):
    def __init__(self, file_dir = '', fps = 3840):
        Game.__init__(self, file_dir)
        self.fps = fps
        self.font = pygame.font.SysFont('Courier New', 30) #Name of font then size
        self.IMAGES['wait'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/added/Wait.png').convert_alpha()
        self.IMAGES['flap'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/added/Flap.png').convert_alpha()
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

    def show_info_cord(self):
        pass

    def _wing_sound(self):
        pass
    
    def _point_sound(self):
        pass
    
    def _hit_sound(self):
        pass
    
    def _die_sound(self):
        pass

class GameManager:   
    def __init__(self, game : Game) -> None:
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
    def set_outputs(self, predict : 'torch.Tensor'):
        '''Will update values that should be displayed during evaluation'''
        self.game.output1 = predict[0].item()
        self.game.output2 = predict[1].item()

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
