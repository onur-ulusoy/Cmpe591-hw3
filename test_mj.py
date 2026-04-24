import numpy as np
from homework4 import Hw5Env

try:
    env = Hw5Env(render_mode="offscreen")
    env.reset()
    state = env.high_level_state()
    print("Env working. State:", state)
except Exception as e:
    print("Error:", e)
