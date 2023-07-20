#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
import os
import struct

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
        short_value = struct.unpack('h', data)[0]

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

for i in range(len(detectors.get_detector_list())):
    r_peaks = detectors.get_detector_list()[i][1](ecg_buffer)

    # convert the sample number to time
    r_ts = np.array(r_peaks) / fs

    if not os.path.isdir(results_dir / detectors.get_detector_list()[i][0]):
        os.makedirs(results_dir / detectors.get_detector_list()[i][0])

    plt.figure(figsize=(200,5))
    t = np.linspace(0, len(ecg_buffer) / fs, len(ecg_buffer))
    plt.plot(t, ecg_buffer)
    plt.plot(r_ts, ecg_buffer[r_peaks], "ro")
    plt.title("Detected R peaks" + detectors.get_detector_list()[i][0])
    plt.ylabel("ECG/mV")
    plt.xlabel("time/sec")
    plt.savefig(results_dir / detectors.get_detector_list()[i][0] / " peaks.png")
    # plt.show()
    plt.close()

    intervals = np.diff(r_ts)
    heart_rate = 60.0 / intervals
    plt.figure(figsize=(1000,1000), dpi=300)
    plt.plot(r_ts[1:], heart_rate)
    plt.title("Heart rate")
    plt.xlabel("time/sec")
    plt.ylabel("HR/BPM")
    plt.savefig(results_dir / detectors.get_detector_list()[i][0] / " heart_rate.png")
    # plt.show()
    plt.close()
