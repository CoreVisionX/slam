# %%
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import load_euroc_sequence_segment

seq = load_euroc_sequence_segment(seq_name="MH_01_easy", sequence_length=150)
frame = seq.frames[-1]

# draw a horizontal line in the left rectified image
left = frame.left.copy()
h_mid = left.shape[0] // 2
left[h_mid-1:h_mid+1, :, :] = 255  # white line

plt.figure()
plt.imshow(frame.left)
plt.title("Left (MH_01_easy)")
plt.show()

plt.figure()
plt.imshow(left)
plt.title("Rectified left (MH_01_easy)")
plt.show()
# %%
