import sounddevice as sd

print(sd.query_devices())
print("default input:", sd.default.device)

import numpy as np


def cb(indata, frames, t, status):
    vol = np.linalg.norm(indata)
    print("audio level:", round(float(vol), 1))

with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1600, callback=cb):
    import time; time.sleep(5)
    print("done")
