#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
import os
import struct
import collections

# max_samples = 2500
# starting_sample = 1000
# ending_sample = starting_sample + max_samples

current_dir = pathlib.Path(__file__).resolve()

example_dir = current_dir.parent / "example_data" / "ECG.tsv"

results_dir = current_dir.parent / "results"

unfiltered_ecg_dat = np.loadtxt(example_dir)
unfiltered_ecg = unfiltered_ecg_dat[:, 0]
fs = 250

file_path = current_dir.parent / "example_data" / "combined.txt"

buffer_size = 2
buffer_one = np.array([], dtype=np.int16)  # Initialize as empty numpy array of shorts
buffer_two = np.array([], dtype=np.int16)  # Initialize as empty numpy array of shorts

active_buffer = 0
with open(file_path, "rb") as file:
    while True:
        data = file.read(buffer_size)

        if not data:
            break

        # Unpack the two bytes into a short (16-bit integer)
        short_value = struct.unpack("h", data)[0]

        # Alternate between buffers
        if active_buffer == 0:
            buffer_one = np.append(buffer_one, short_value)
            active_buffer = 1
        else:
            buffer_two = np.append(buffer_two, short_value)
            active_buffer = 0

detectors = Detectors(fs)

# ecg_buffer = buffer_one
ecg_buffer = buffer_two
# ecg_buffer = unfiltered_ecg

filtered_buffer = np.array([], dtype=np.int16)
avg_buffer = np.array([], dtype=np.int16)


# # Quick averager
# newvalue = 0
# total = 0
# count = 0
max_count = 5
# # max_count = 250 * 60
# # tmp_buffer = np.array([], dtype=np.int16)
# # tmp_buffer = np.append(tmp_buffer, 0)

# tmp_buffer = collections.deque(maxlen = max_count)
# # tmp_buffer.appendleft(0)

# for sample in ecg_buffer:
#     popped_value = 0

#     if count >= max_count:
#         popped_value = tmp_buffer.popleft()
#     else:
#         count += 1
#     total += sample - popped_value

#     tmp_buffer.append(sample)

#     avg = total / count

#     filtered_value = sample - avg

#     filtered_buffer = np.append(filtered_buffer, filtered_value)


# # Rolling Average w/ accumulator
# accumulator = 0
# tempAvg = 0
# for sample in ecg_buffer:

#     if count == 0:
#         accumulator = sample
#         count += 1
#     else:
#         tempAvg = accumulator
#         # Remove oldest value
#         tempAvg -= accumulator / count
#         # Add newest value
#         tempAvg += sample / count
#         if count < max_count:
#             count += 1
#         accumulator = tempAvg

#     filtered_buffer = np.append(filtered_buffer, sample - accumulator)

# Exponentially weighted moving average alternative
count = 0
old_avg = 0
new_avg = 0
max_count = 1024
for sample in ecg_buffer:

    if count < max_count:
      count += 1

    if count == 1:
        old_avg = new_avg = sample
    else:
        new_avg = old_avg + (sample - old_avg) / count
        old_avg = new_avg

    filtered_buffer = np.append(filtered_buffer, sample - new_avg)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

plt.figure(figsize=(200, 5))
t = np.linspace(0, len(filtered_buffer) / fs, len(filtered_buffer))
plt.plot(t, filtered_buffer)
plt.title("Filtered ECG")
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.savefig(results_dir / f"filtered({str(max_count)}).png")
# plt.show()
plt.close()

# for i in range(len(detectors.get_detector_list())):
#     r_peaks = detectors.get_detector_list()[i][1](ecg_buffer)

#     # convert the sample number to time
#     r_ts = np.array(r_peaks) / fs

#     if not os.path.isdir(results_dir / detectors.get_detector_list()[i][0]):
#         os.makedirs(results_dir / detectors.get_detector_list()[i][0])

#     plt.figure(figsize=(200,5))
#     t = np.linspace(0, len(ecg_buffer) / fs, len(ecg_buffer))
#     plt.plot(t, ecg_buffer)
#     plt.plot(r_ts, ecg_buffer[r_peaks], "ro")
#     plt.title("Detected R peaks" + detectors.get_detector_list()[i][0])
#     plt.ylabel("ECG/mV")
#     plt.xlabel("time/sec")
#     plt.savefig(results_dir / detectors.get_detector_list()[i][0] / " peaks.png")
#     # plt.show()
#     plt.close()

#     intervals = np.diff(r_ts)
#     heart_rate = 60.0 / intervals
#     plt.figure(figsize=(1000,1000), dpi=300)
#     plt.plot(r_ts[1:], heart_rate)
#     plt.title("Heart rate")
#     plt.xlabel("time/sec")
#     plt.ylabel("HR/BPM")
#     plt.savefig(results_dir / detectors.get_detector_list()[i][0] / " heart_rate.png")
#     # plt.show()
#     plt.close()
