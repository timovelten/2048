from blessed import Terminal
import twentyfortyeight as twfe
import numpy as np

# Interactive terminal based 2048, to make sure that we have implemented the game correctly
# (surprisingly hard...)

def play():
    rng = np.random.default_rng()
    state = twfe.State.starting_state(rng)
    term = Terminal()

    while True:
        print(str(state))
        if state.is_terminated:
            print("Game over")
            break

        round_done = False
        while not round_done:
            with term.raw():
                inp = term.inkey()
                if inp == chr(3):
                    exit()

            if inp == "w" or inp.code == term.KEY_UP:
                delta = state.next_states[twfe.UP]
            elif inp == "d" or inp.code == term.KEY_DOWN:
                delta = state.next_states[twfe.DOWN]
            elif inp == "a" or inp.code == term.KEY_LEFT:
                delta = state.next_states[twfe.LEFT]
            elif inp == "d" or inp.code == term.KEY_RIGHT:
                delta = state.next_states[twfe.RIGHT]

            if delta.legal:
                round_done = True
                state = delta.state
                state.add_random_tile(rng)

if __name__ == "__main__":
    play()