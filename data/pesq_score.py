import re
import subprocess
import librosa
import os
import numpy as np
from scipy import interpolate
import soundfile as sf


regex = re.compile(r'P\.862\.2 Prediction \(MOS-LQO\):  = (\d+\.\d+)')
pesq_path = 'PESQ/pesq'
pesq_sr = 16000


temp_dir = os.path.join(os.getcwd(), 'temp')
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)


def run_pesq(reference_file, degraded_file):
    command = ['./' + pesq_path, reference_file, degraded_file, '+16000', '+wb']
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return result.stdout.decode()


def get_pesq(reference_file, degraded_file):

    output = run_pesq(reference_file, degraded_file)

    matches = regex.findall(output)
    assert len(matches) == 1, "No score: " + output
    score = float(matches[0])
    assert score > 0.0 and score < 5.0, "Score not valid: " + str(score)

    return score
