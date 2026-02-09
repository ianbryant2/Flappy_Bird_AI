from itertools import cycle
from pathlib import Path
import random
import sys
import pygame
from pygame.locals import *

xrange = range


class Game:
    def __init__(self, file_dir: Path = Path("./")):
        self.screen_width = 288
        self.screen_height = 512
        self.pipe_gap_size = 100  # gap between upper and lower part of pipe
        self.base_y = self.screen_height * 0.79

        # image, sound and hitmask dicts
        self.images, self.sounds, self.hitmasks = {}, {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.players_list = (
            (
                file_dir / Path("FlappyBird/assets/sprites/redbird-upflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/redbird-midflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/redbird-downflap.png"),
            ),
            (
                file_dir / Path("FlappyBird/assets/sprites/bluebird-upflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/bluebird-midflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/bluebird-downflap.png"),
            ),
            (
                file_dir / Path("FlappyBird/assets/sprites/yellowbird-upflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/yellowbird-midflap.png"),
                file_dir / Path("FlappyBird/assets/sprites/yellowbird-downflap.png"),
            ),
        )

        # list of backgrounds
        self.backgrounds_list = (
            file_dir / Path("FlappyBird/assets/sprites/background-day.png"),
            file_dir / Path("FlappyBird/assets/sprites/background-night.png"),
        )

        # list of pipes
        self.pipes_list = (
            file_dir / Path("FlappyBird/assets/sprites/pipe-green.png"),
            file_dir / Path("FlappyBird/assets/sprites/pipe-red.png"),
        )

        self.xrange = range

        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flappy Bird")

        # numbers sprites for score display
        self.images["numbers"] = tuple(
            pygame.image.load(file_dir / Path(f"FlappyBird/assets/sprites/{i}.png")).convert_alpha()
            for i in range(10)
        )

        # game over sprite
        self.images["gameover"] = pygame.image.load(
            file_dir / Path("FlappyBird/assets/sprites/gameover.png")
        ).convert_alpha()

        # message sprite for welcome screen
        self.images["message"] = pygame.image.load(
            file_dir / Path("FlappyBird/assets/sprites/message.png")
        ).convert_alpha()

        # base (ground) sprite
        self.images["base"] = pygame.image.load(
            file_dir / Path("FlappyBird/assets/sprites/base.png")
        ).convert_alpha()

        # sounds
        self.sound_ext = ".wav" if "win" in sys.platform else ".ogg"

        self.sounds["die"] = pygame.mixer.Sound(file_dir / Path("FlappyBird/assets/audio/die" + self.sound_ext))
        self.sounds["hit"] = pygame.mixer.Sound(file_dir / Path("FlappyBird/assets/audio/hit" + self.sound_ext))
        self.sounds["point"] = pygame.mixer.Sound(file_dir / Path("FlappyBird/assets/audio/point" + self.sound_ext))
        self.sounds["swoosh"] = pygame.mixer.Sound(file_dir / Path("FlappyBird/assets/audio/swoosh" + self.sound_ext))
        self.sounds["wing"] = pygame.mixer.Sound(file_dir / Path("FlappyBird/assets/audio/wing" + self.sound_ext))

    def init_level(self):
        """used to start initialize the level"""
        # select random background sprites
        self.rand_bg = random.randint(0, len(self.backgrounds_list) - 1)
        self.images["background"] = pygame.image.load(self.backgrounds_list[self.rand_bg]).convert()

        # select random player sprites
        self.rand_player = random.randint(0, len(self.players_list) - 1)
        self.images["player"] = (
            pygame.image.load(self.players_list[self.rand_player][0]).convert_alpha(),
            pygame.image.load(self.players_list[self.rand_player][1]).convert_alpha(),
            pygame.image.load(self.players_list[self.rand_player][2]).convert_alpha(),
        )

        # select random pipe sprites
        self.pipe_index = random.randint(0, len(self.pipes_list) - 1)
        self.images["pipe"] = (
            pygame.transform.flip(
                pygame.image.load(self.pipes_list[self.pipe_index]).convert_alpha(),
                False,
                True,
            ),
            pygame.image.load(self.pipes_list[self.pipe_index]).convert_alpha(),
        )

        # hitmask for pipes
        self.hitmasks["pipe"] = (
            self.get_hitmask(self.images["pipe"][0]),
            self.get_hitmask(self.images["pipe"][1]),
        )

        # hitmask for player
        self.hitmasks["player"] = (
            self.get_hitmask(self.images["player"][0]),
            self.get_hitmask(self.images["player"][1]),
            self.get_hitmask(self.images["player"][2]),
        )

        # welcome screen animation state
        self.player_index = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        self.loop_iter = 0

        self.player_x = int(self.screen_width * 0.2)
        self.player_y = int((self.screen_height - self.images["player"][0].get_height()) / 2)

        self.message_x = int((self.screen_width - self.images["message"].get_width()) / 2)
        self.message_y = int(self.screen_height * 0.12)

        self.base_x = 0
        self.base_shift = self.images["base"].get_width() - self.images["background"].get_width()

        # player shm for up-down motion on welcome screen
        self.player_shm_vals = {"val": 0, "dir": 1}
        self.movement_info = self.show_welcome_animation()

        self.score = self.player_index = self.loop_iter = 0
        self.player_index_gen = self.movement_info["player_index_gen"]
        self.player_x, self.player_y = int(self.screen_width * 0.2), self.movement_info["player_y"]
        self.base_x = self.movement_info["base_x"]
        self.base_shift = self.images["base"].get_width() - self.images["background"].get_width()

        # get 2 new pipes to add to upper_pipes lower_pipes list
        self.new_pipe_1 = self.get_random_pipe()
        self.new_pipe_2 = self.get_random_pipe()

        self.upper_pipes = [
            {"x": self.screen_width + 200, "y": self.new_pipe_1[0]["y"]},
            {"x": self.screen_width + 200 + (self.screen_width / 2), "y": self.new_pipe_2[0]["y"]},
        ]

        self.lower_pipes = [
            {"x": self.screen_width + 200, "y": self.new_pipe_1[1]["y"]},
            {"x": self.screen_width + 200 + (self.screen_width / 2), "y": self.new_pipe_2[1]["y"]},
        ]

        self.pipe_vel_x = -4

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.player_vel_y = -9
        self.player_max_vel_y = 10
        self.player_min_vel_y = -8
        self.player_acc_y = 1
        self.player_rot = 45
        self.player_vel_rot = 3
        self.player_rot_thr = 20
        self.player_flap_acc = -9
        self.player_flapped = False

    def show_welcome_animation(self):
        """shows the welcome animation"""
        while True:
            values = self.intro_looper()

            # adjust player_y, player_index, base_x
            if (self.loop_iter + 1) % 5 == 0:
                self.player_index = next(self.player_index_gen)
            self.loop_iter = (self.loop_iter + 1) % 30
            self.base_x = -((-self.base_x + 4) % self.base_shift)
            self.player_shm(self.player_shm_vals)

            # draw sprites
            self.screen.blit(self.images["background"], (0, 0))
            self.screen.blit(
                self.images["player"][self.player_index],
                (self.player_x, self.player_y + self.player_shm_vals["val"]),
            )
            self.screen.blit(self.images["message"], (self.message_x, self.message_y))
            self.screen.blit(self.images["base"], (self.base_x, self.base_y))

            pygame.display.update()
            self.fps_clock.tick(self.get_fps())

            if values is not None:
                return values

    def level_loop(self):
        """completes one game loop"""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.get_inputs()):
                if self.player_y > -2 * self.images["player"][0].get_height():
                    self.player_vel_y = self.player_flap_acc
                    self.player_flapped = True
                    self.wing_sound()

        # check for crash here
        crash_test = self.check_crash(
            {"x": self.player_x, "y": self.player_y, "index": self.player_index},
            self.upper_pipes,
            self.lower_pipes,
        )
        if crash_test[0]:
            return {
                "y": self.player_y,
                "ground_crash": crash_test[1],
                "base_x": self.base_x,
                "upper_pipes": self.upper_pipes,
                "lower_pipes": self.lower_pipes,
                "score": self.score,
                "player_vel_y": self.player_vel_y,
                "player_rot": self.player_rot,
            }

        # check for score
        player_mid_pos = self.player_x + self.images["player"][0].get_width() / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe["x"] + self.images["pipe"][0].get_width() / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                self.point_sound()

        # player_index, base_x change
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
            self.player_rot = 45

        player_height = self.images["player"][self.player_index].get_height()
        self.player_y += min(self.player_vel_y, self.base_y - self.player_y - player_height)

        # move pipes to left
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe["x"] += self.pipe_vel_x
            lower_pipe["x"] += self.pipe_vel_x

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]["x"] < 5:
            new_pipe = self.get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if len(self.upper_pipes) > 0 and self.upper_pipes[0]["x"] < -self.images["pipe"][0].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # draw sprites
        self.screen.blit(self.images["background"], (0, 0))

        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.images["pipe"][0], (upper_pipe["x"], upper_pipe["y"]))
            self.screen.blit(self.images["pipe"][1], (lower_pipe["x"], lower_pipe["y"]))

        self.screen.blit(self.images["base"], (self.base_x, self.base_y))

        # print score so player overlaps the score
        self.show_score(self.score)

        # Player rotation has a threshold
        visible_rot = self.player_rot_thr
        if self.player_rot <= self.player_rot_thr:
            visible_rot = self.player_rot

        player_surface = pygame.transform.rotate(self.images["player"][self.player_index], visible_rot)
        self.screen.blit(player_surface, (self.player_x, self.player_y))

        self.show_info()
        pygame.display.update()
        self.fps_clock.tick(self.get_fps())

        return None  # keep consistent return behavior

    def check_crash(self, player, upper_pipes, lower_pipes):
        """returns True if player collides with base or pipes."""
        pi = player["index"]
        player["w"] = self.images["player"][0].get_width()
        player["h"] = self.images["player"][0].get_height()

        # if player crashes into ground
        if player["y"] + player["h"] >= self.base_y - 1:
            return [True, True]

        player_rect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
        pipe_w = self.images["pipe"][0].get_width()
        pipe_h = self.images["pipe"][0].get_height()

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            upper_pipe_rect = pygame.Rect(upper_pipe["x"], upper_pipe["y"], pipe_w, pipe_h)
            lower_pipe_rect = pygame.Rect(lower_pipe["x"], lower_pipe["y"], pipe_w, pipe_h)

            p_hitmask = self.hitmasks["player"][pi]
            u_hitmask = self.hitmasks["pipe"][0]
            l_hitmask = self.hitmasks["pipe"][1]

            upper_collide = self.pixel_collision(player_rect, upper_pipe_rect, p_hitmask, u_hitmask)
            lower_collide = self.pixel_collision(player_rect, lower_pipe_rect, p_hitmask, l_hitmask)

            if upper_collide or lower_collide:
                return [True, False]

        return [False, False]

    def show_game_over_screen(self, crash_info):
        """crashes the player down and shows gameover image"""
        score = crash_info["score"]
        player_x = self.screen_width * 0.2
        player_y = crash_info["y"]
        player_height = self.images["player"][0].get_height()
        player_vel_y = crash_info["player_vel_y"]
        player_acc_y = 2
        player_rot = crash_info["player_rot"]
        player_vel_rot = 7

        self.base_x = crash_info["base_x"]

        upper_pipes, lower_pipes = crash_info["upper_pipes"], crash_info["lower_pipes"]

        # play hit and die sounds
        self.hit_sound()
        if not crash_info["ground_crash"]:
            self.die_sound()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key in self.get_inputs()):
                    if player_y + player_height >= self.base_y - 1:
                        return

            # player y shift
            if player_y + player_height < self.base_y - 1:
                player_y += min(player_vel_y, self.base_y - player_y - player_height)

            # player velocity change
            if player_vel_y < 15:
                player_vel_y += player_acc_y

            # rotate only when it's a pipe crash
            if not crash_info["ground_crash"]:
                if player_rot > -90:
                    player_rot -= player_vel_rot

            # draw sprites
            self.screen.blit(self.images["background"], (0, 0))

            for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
                self.screen.blit(self.images["pipe"][0], (upper_pipe["x"], upper_pipe["y"]))
                self.screen.blit(self.images["pipe"][1], (lower_pipe["x"], lower_pipe["y"]))

            self.screen.blit(self.images["base"], (self.base_x, self.base_y))
            self.show_score(score)

            player_surface = pygame.transform.rotate(self.images["player"][1], player_rot)
            self.screen.blit(player_surface, (player_x, player_y))
            self.screen.blit(self.images["gameover"], (50, 180))

            self.fps_clock.tick(self.get_fps())
            pygame.display.update()

    def show_score(self, score):
        """displays score in center of screen"""
        score_digits = [int(x) for x in list(str(score))]
        total_width = 0

        for digit in score_digits:
            total_width += self.images["numbers"][digit].get_width()

        x_offset = (self.screen_width - total_width) / 2

        for digit in score_digits:
            self.screen.blit(self.images["numbers"][digit], (x_offset, self.screen_height * 0.1))
            x_offset += self.images["numbers"][digit].get_width()

    def get_random_pipe(self):
        """returns a randomly generated pipe"""
        gap_y = random.randrange(0, int(self.base_y * 0.6 - self.pipe_gap_size))
        gap_y += int(self.base_y * 0.2)
        pipe_height = self.images["pipe"][0].get_height()
        pipe_x = self.screen_width + 10

        return [
            {"x": pipe_x, "y": gap_y - pipe_height},          # upper pipe
            {"x": pipe_x, "y": gap_y + self.pipe_gap_size},   # lower pipe
        ]

    def player_shm(self, player_shm):
        """oscillates the value of player_shm['val'] between 8 and -8"""
        if abs(player_shm["val"]) == 8:
            player_shm["dir"] *= -1
        if player_shm["dir"] == 1:
            player_shm["val"] += 1
        else:
            player_shm["val"] -= 1

    def pixel_collision(self, rect_1, rect_2, hitmask_1, hitmask_2):
        """Checks if two objects collide and not just their rects"""
        rect = rect_1.clip(rect_2)
        if rect.width == 0 or rect.height == 0:
            return False

        x_1, y_1 = rect.x - rect_1.x, rect.y - rect_1.y
        x_2, y_2 = rect.x - rect_2.x, rect.y - rect_2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask_1[x_1 + x][y_1 + y] and hitmask_2[x_2 + x][y_2 + y]:
                    return True
        return False

    def get_hitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in xrange(image.get_width()):
            mask.append([])
            for y in xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def show_info(self):
        pass

    def get_fps(self):
        """Returns the FPS for current game"""
        return 30

    def get_inputs(self):
        """Returns the correct input for current game"""
        return [K_SPACE, K_UP]

    def intro_looper(self):
        """First game loop in intro screen"""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.get_inputs()):
                self.wing_sound()
                return {
                    "player_y": self.player_y + self.player_shm_vals["val"],
                    "base_x": self.base_x,
                    "player_index_gen": self.player_index_gen,
                }

    def wing_sound(self):
        self.sounds["wing"].play()

    def point_sound(self):
        self.sounds["point"].play()

    def hit_sound(self):
        self.sounds["hit"].play()

    def die_sound(self):
        self.sounds["die"].play()

    def assign_action(self):
        pass

    def assign_wait(self):
        pass


class PlayGame(Game):
    def __init__(self, file_dir=Path("./")):
        super().__init__(file_dir=file_dir)

    def get_fps(self):
        return 30

    def get_inputs(self):
        return [K_SPACE, K_UP]

    def intro_looper(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.get_inputs()):
                self.wing_sound()
                return {
                    "player_y": self.player_y + self.player_shm_vals["val"],
                    "base_x": self.base_x,
                    "player_index_gen": self.player_index_gen,
                }


class TrainGame(Game):
    def __init__(self, file_dir=Path("./")):
        super().__init__(file_dir=file_dir)

    def get_fps(self):
        return 3840

    def get_inputs(self):
        return [K_AC_BACK]  # Using Android Backspace Key because not on PC keyboard

    def intro_looper(self):
        return {
            "player_y": self.player_y + self.player_shm_vals["val"],
            "base_x": self.base_x,
            "player_index_gen": self.player_index_gen,
        }

    def wing_sound(self):
        pass

    def point_sound(self):
        pass

    def hit_sound(self):
        pass

    def die_sound(self):
        pass


class EvaluateGame(Game):
    def __init__(self, file_dir: Path = Path("./"), fps=3840):
        super().__init__(file_dir=file_dir)
        self.fps = fps
        self.font = pygame.font.SysFont("Courier New", 30)

        self.images["wait"] = pygame.image.load(
            file_dir / Path("FlappyBird/assets/sprites/added/Wait.png")
        ).convert_alpha()
        self.images["flap"] = pygame.image.load(
            file_dir / Path("FlappyBird/assets/sprites/added/Flap.png")
        ).convert_alpha()

        self.output_1 = None
        self.output_2 = None
        self.flap_count = 0
        self.image_shown = None

    def get_fps(self):
        return self.fps

    def get_inputs(self):
        return [K_AC_BACK]

    def intro_looper(self):
        return {
            "player_y": self.player_y + self.player_shm_vals["val"],
            "base_x": self.base_x,
            "player_index_gen": self.player_index_gen,
        }

    def assign_action(self):
        self.image_shown = self.images["flap"]
        self.flap_count = 1

    def assign_wait(self):
        if self.flap_count > 2:
            self.flap_count = 0
            self.image_shown = self.images["wait"]
        else:
            self.flap_count += 1

    def show_info(self):
        if self.image_shown is not None:
            self.screen.blit(self.image_shown, (150, 425))
            text_output_1 = self.font.render(str(int(self.output_1)), True, (0, 0, 0))
            text_output_2 = self.font.render(str(int(self.output_2)), True, (0, 0, 0))
            self.screen.blit(text_output_1, (10, 435))
            self.screen.blit(text_output_2, (10, 475))

    def show_info_cord(self):
        pass

    def wing_sound(self):
        pass

    def point_sound(self):
        pass

    def hit_sound(self):
        pass

    def die_sound(self):
        pass


class GameManger:
    def __init__(self, game: Game, fps_count=1):
        # fps_count is number of frames between when the reward is returned after the action
        self.movement = None
        self.score_check = 0
        self.fps_count = fps_count
        self.game = game
        self.key_event_up = pygame.event.Event(KEYDOWN, {"key": self.game.get_inputs()[0]})

        self.top_count = 0
        self.bottom_count = 0
        self.upper_dist_offset = 10
        self.lower_dist_offset = 20

    def execute_action(self):
        if self.movement == [1, 0]:
            pygame.event.post(self.key_event_up)
            self.game.assign_action()
        else:
            self.game.assign_wait()

    def action(self, action, score):
        self.movement = action
        self.score_check = score
        self.execute_action()

    def determine_pos_reward(self):
        return 8.88889 * (self.upper_dist_offset) + 355.556

    def get_reward(self, crash_info, score):
        reward = 0
        done = False

        distance_top = self.game.player_y - (
            self.game.images["pipe"][0].get_height() + self.game.upper_pipes[-2]["y"]
        )
        distance_bottom = self.game.lower_pipes[-2]["y"] - self.game.player_y

        if distance_top < self.upper_dist_offset or distance_bottom < self.lower_dist_offset:
            reward += -200
        else:
            reward += self.determine_pos_reward()

        if crash_info is not None:
            done = True
            reward += -200
            return reward, done, score
        if score > self.score_check:
            reward += 200
            return reward, done, score

        return reward, done, score

    def action_sequence(self, action):
        self.action(action, self.game.score)
        crash_info = None
        for _ in range(self.fps_count):
            crash_info = self.game.level_loop()
        return self.get_reward(crash_info, self.game.score)

    def reset(self):
        self.game.init_level()
        pygame.event.post(self.key_event_up)

    def get_state(self):
        try:
            distance_top_first = self.game.player_y - (
                self.game.images["pipe"][0].get_height() + self.game.upper_pipes[-2]["y"]
            )
            distance_top_second = self.game.player_y - (
                self.game.images["pipe"][0].get_height() + self.game.upper_pipes[-1]["y"]
            )
            return [
                self.game.lower_pipes[-2]["x"],
                self.game.lower_pipes[-1]["x"],
                distance_top_first,
                distance_top_second,
                self.game.player_vel_y,
            ]
        except AttributeError:
            return (0, 0, 0, 0, 0)

    def play(self):
        while True:
            self.game.init_level()
            crash_info = None
            while crash_info is None:
                crash_info = self.game.level_loop()
            self.game.show_game_over_screen(crash_info)

    def set_outputs(self, predict):
        # predict is a tensor
        self.game.output_1 = predict[0].item()
        self.game.output_2 = predict[1].item()
