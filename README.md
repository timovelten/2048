This is my attempt at using reinforcement learning to solve the game [2048](https://play2048.co/).

It features:
- A `gym.Env` to play 2048, where performance critical parts are numba compiled (this is important, as most of the training time is actually spent simulating environments)
- A custom somewhat generic PPO implementation (loosely based on the stable baselines 3 implementation)
- Action masking. In my testing, the agent really struggled to learn when a move was legal, and it really hindered progress.
- A simple reward function. A natural choice of reward function for 2048 is to simply award the agent the score delta between two states (i.e. merging two 1024 tiles to give one 2048 tile would give a reward of 2048.0). However, this did not work very well in practice. I assume this is due to the exponential nature of these rewards. The reward function that I currently use is `(score delta) / (sum of all tiles on the board)`, and the agent is awarded a small bonus for actually achiving the 2048 tile. Critically, I do not reward any hand picked features (for example, giving a reward for putting larger tiles into the corners).

### Training

With the current hyper parameters, training is rather slow. It takes around `70 000 000` timesteps for the agent to somewhat consistently achieve the 2048 tile (around `15%` of the time in validation). Training for a lot longer (around `800 000 000` timesteps, which took 10 hours), the agent finally achieves the 2048 tile in `75%` of attempts, and achieves the `4096` tile in around `15%` of cases.

### Demo

Watch the model achive the 4096 tile (using the weights stored under `weights/`) here. You can watch it play in your own browser by following the instructions in `misc/play_browser.py`.

[demo.webm](https://github.com/user-attachments/assets/621af019-e2df-4d38-8f33-846aaa29944e)
