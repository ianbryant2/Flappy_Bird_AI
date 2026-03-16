
from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
from dqn import Agent, train, evaluate
import torch
import os
import argparse

file_dir = os.path.dirname(os.path.realpath(__file__))


def main():
   parser = argparse.ArgumentParser(description='Flappy Bird AI')
   parser.add_argument(
      'mode',
      choices=['play', 'train', 'test', 'evaluate'],
      type=str.lower,
      help='Mode to run the game in',
   )
   parser.add_argument(
      '--fps',
      type=int,
      default=30,
      help='Frames per second (default: 30)',
   )
   parser.add_argument(
      '--epochs',
      type=lambda x: None if x.lower() == 'infinite' else int(x),
      default=None,
      help='Number of epochs, or "infinite" (default: infinite for evaluate, 5000 otherwise)',
   )
   args = parser.parse_args()

   epochs = args.epochs
   if epochs is None and args.mode != 'evaluate':
      epochs = 5000

   if args.mode == 'train':
      gm = fp.GameManger(TrainGame(file_dir=file_dir), fps_count=3)
      agent = Agent(gm)
      train(agent, epochs=epochs, plotting_scores=True)

   elif args.mode == 'play':
      gm = fp.GameManger(PlayGame(file_dir=file_dir))
      gm.play()

   elif args.mode == 'evaluate':
      gm = fp.GameManger(EvaluateGame(file_dir=file_dir, fps=args.fps))
      agent = Agent(gm)
      agent.model.load_state_dict(torch.load(file_dir + '/test_weights.pt'))
      evaluate(agent, epochs=epochs)

   elif args.mode == 'test':
      gm = fp.GameManger(TrainGame(file_dir=file_dir))
      agent = Agent(gm)
      train(agent, epochs=epochs)
      gm = fp.GameManger(EvaluateGame(file_dir=file_dir))
      agent.new_game_manager(gm)
      print('starting to evaluate')
      for i in range(3):
         evaluate(agent, run_num=i, epochs=250)


if __name__ == "__main__":
   main()

