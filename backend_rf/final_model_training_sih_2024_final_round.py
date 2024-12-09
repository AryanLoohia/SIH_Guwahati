import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Example data (assuming you have a DataFrame with the extracted features)
df1 = pd.DataFrame(pd.read_excel('Features_Training_Initial_55_07_12_2024.xlsx'))
df21 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_1.xlsx'))
df22 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_2.xlsx'))
df23 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_3.xlsx'))
df24 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_4.xlsx'))
df25 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_5.xlsx'))
df26 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_07_12_2024_6.xlsx'))
df2 = pd.concat([df21, df22, df23, df24, df25, df26], axis=0)

df1

df2

df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)
print(common_columns)

df2 = df2.reset_index(drop=True)
df2

df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)', 'Encrypted Data (Binary)', 'Original Text', 'Length', 'Algorithm'], inplace=True)
df = pd.concat([df1, df2], axis=1)
df

"""#Engineering Some Other Features"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_block_size(ciphertext_hex):
   
    ciphertext_bytes = bytes.fromhex(ciphertext_hex)

    # Step 2: Get the length of the ciphertext in bytes
    ciphertext_length = len(ciphertext_bytes)

    # Step 3: Check the length of the ciphertext and infer the block size
    # print(f"Ciphertext length in bytes: {ciphertext_length}")

    if ciphertext_length % 16 == 0:
        return 16
    elif ciphertext_length % 8 == 0:
        return 8
    else:
        return 0

# Function to calculate the block frequency (check repetition)
def calculate_block_frequency(ciphertext, block_size=8):
    blocks = [ciphertext[i:i+block_size] for i in range(0, len(ciphertext), block_size)]
    block_counter = Counter(tuple(block) for block in blocks)

    # Calculate the percentage of repeated blocks
    repeated_blocks = sum(count > 1 for count in block_counter.values())
    total_blocks = len(blocks)
    return repeated_blocks / total_blocks * 100

block_size_list = []
block_cipher_boolean_list = []
block_frequency_list = []

for i in df['Encrypted Data (Hex)']:
    block_size = get_block_size(i)
    block_size_list.append(block_size)
    if (block_size != 0):
        block_cipher_boolean_list.append(1)
    else:
        block_cipher_boolean_list.append(0)
    if (block_size == 8):
        block_frequency_list.append(calculate_block_frequency(i, block_size))
    else:
        block_frequency_list.append(0)
df['block_size'] = block_size_list
df['block_cipher_boolean'] = block_cipher_boolean_list
df['block_frequency'] = block_frequency_list

import numpy as np
import math
import itertools
import pandas as pd
from scipy import special

# Function to convert Hex to Binary
def hex_to_binary(hex_data):
    return ''.join(format(byte, '08b') for byte in bytes.fromhex(hex_data))

# Frequency Test (Monobit Test)
def frequency_test(binary_sequence):
    """
    Checks the proportion of 0s and 1s in the binary sequence

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    n = len(binary_sequence)
    s = np.sum([int(bit) for bit in binary_sequence])
    s_obs = abs(s - n/2) / np.sqrt(n/4)

    # Calculate p-value
    p_value = math.erfc(s_obs / np.sqrt(2))
    return p_value

# Runs Test
def runs_test(binary_sequence):
    """
    Analyzes the number of runs (consecutive sequences of the same bit)

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    n = len(binary_sequence)
    pi = np.mean([int(bit) for bit in binary_sequence])

    # Calculate the number of runs
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1

    # Calculate test statistic
    runs_obs = runs
    runs_exp = ((2 * n * pi * (1 - pi)) + 1)
    runs_std = 2 * np.sqrt(2 * n * pi * (1 - pi)) - (1/2)

    z = (runs_obs - runs_exp) / runs_std

    # Calculate p-value
    p_value = math.erfc(abs(z) / np.sqrt(2))
    return p_value

# Longest Run of Ones Test
def longest_run_test(binary_sequence):
    """
    Checks the longest run of consecutive 1s in the sequence

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    # Split sequence into blocks
    k = 6  # Number of blocks
    m = 8  # Length of each block

    blocks = [binary_sequence[i:i+m] for i in range(0, len(binary_sequence), m)]

    # Count longest runs in each block
    longest_runs = []
    for block in blocks:
        max_run = 0
        current_run = 0

        # Find the longest run of 1s in the block
        for bit in block:
            if bit == '1':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        longest_runs.append(max_run)

    # Predefined chi-square distribution parameters
    v_obs = [0, 0, 0, 0]
    for run in longest_runs:
        if run <= 1:
            v_obs[0] += 1
        elif run == 2:
            v_obs[1] += 1
        elif run == 3:
            v_obs[2] += 1
        elif run >= 4:
            v_obs[3] += 1

    # Predefined probabilities for chi-square
    pi = [0.2148, 0.3672, 0.2305, 0.1875]

    # Chi-square calculation
    chi_sq = sum(((v_obs[i] - k * pi[i])**2) / (k * pi[i]) for i in range(4))

    # Calculate p-value
    p_value = special.gammainc(2.5, chi_sq/2)
    return p_value

# Function to calculate cryptographic features
def extract_features(hex_data):
    binary_data = hex_to_binary(hex_data)

    # Perform NIST randomness tests
    freq_p_value = frequency_test(binary_data)
    runs_p_value = runs_test(binary_data)
    longest_run_p_value = longest_run_test(binary_data)



    # Combine all the extracted features into a dictionary
    features = {
        'nist_frequency_test_p_value': freq_p_value,
        'nist_runs_test_p_value': runs_p_value,
        'nist_longest_run_test_p_value': longest_run_p_value,

    }

    return features

# Assuming 'Encrypted Data (Hex)' column exists in the DataFrame
nist_p_values_list = []
nist_p_values_runs_test_list = []
nist_p_values_longest_run_list = []

for i in df['Encrypted Data (Hex)']:
    features = extract_features(i)

    nist_p_values_list.append(features['nist_frequency_test_p_value'])
    nist_p_values_runs_test_list.append(features['nist_runs_test_p_value'])
    nist_p_values_longest_run_list.append(features['nist_longest_run_test_p_value'])


# Add the results to the dataframe
df['nist_frequency_test_p_value'] = nist_p_values_list
df['nist_runs_test_p_value'] = nist_p_values_runs_test_list
df['nist_longest_run_test_p_value'] = nist_p_values_longest_run_list

df

count = 0
total = 0
for i in range(0, len(df['Algorithm'])):
    if (df['Algorithm'][i] == 'AES'):
        total = total + 1
        # print(df['block_size'][i])
        if (df['block_size'][i] == 16):
            count = count + 1
print(count / total)

"""#Preparing Training Dataset"""

y_train = df['Algorithm']
df.drop(columns=['Original Text', 'Length', 'Encrypted Data (Binary)', 'Encrypted Data (Hex)', 'Algorithm', 'iv'], inplace=True)
X_train = df

y_train

X_train

"""##**Input (Testing)**

"""

# Example data (assuming you have a DataFrame with the extracted features)
df1 = pd.DataFrame(pd.read_excel('Features_Testing_Initial_55_07_12_2024.xlsx'))
df2 = pd.DataFrame(pd.read_excel('Features_Testing_Final_30_07_12_2024.xlsx'))
df1

df2

df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)
print(common_columns)

df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)', 'Encrypted Data (Binary)', 'Original Text', 'Length', 'Algorithm'], inplace=True)
df = pd.concat([df1, df2], axis=1)
df

"""#Engineering Some Other Features"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_block_size(ciphertext_hex):
    # Step 1: Convert the hex string to bytes
    ciphertext_bytes = bytes.fromhex(ciphertext_hex)

    # Step 2: Get the length of the ciphertext in bytes
    ciphertext_length = len(ciphertext_bytes)

    # Step 3: Check the length of the ciphertext and infer the block size
    # print(f"Ciphertext length in bytes: {ciphertext_length}")

    if ciphertext_length % 16 == 0:
        return 16
    elif ciphertext_length % 8 == 0:
        return 8
    else:
        return 0

# Function to calculate the block frequency (check repetition)
def calculate_block_frequency(ciphertext, block_size=8):
    blocks = [ciphertext[i:i+block_size] for i in range(0, len(ciphertext), block_size)]
    block_counter = Counter(tuple(block) for block in blocks)

    # Calculate the percentage of repeated blocks
    repeated_blocks = sum(count > 1 for count in block_counter.values())
    total_blocks = len(blocks)
    return repeated_blocks / total_blocks * 100

block_size_list = []
block_cipher_boolean_list = []
block_frequency_list = []

for i in df['Encrypted Data (Hex)']:
    block_size = get_block_size(i)
    block_size_list.append(block_size)
    if (block_size != 0):
        block_cipher_boolean_list.append(1)
    else:
        block_cipher_boolean_list.append(0)
    if (block_size == 8):
        block_frequency_list.append(calculate_block_frequency(i, block_size))
    else:
        block_frequency_list.append(0)
df['block_size'] = block_size_list
df['block_cipher_boolean'] = block_cipher_boolean_list
df['block_frequency'] = block_frequency_list

import numpy as np
import math
import itertools
import pandas as pd
from scipy import special

# Function to convert Hex to Binary
def hex_to_binary(hex_data):
    return ''.join(format(byte, '08b') for byte in bytes.fromhex(hex_data))

# Frequency Test (Monobit Test)
def frequency_test(binary_sequence):
    """
    Checks the proportion of 0s and 1s in the binary sequence

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    n = len(binary_sequence)
    s = np.sum([int(bit) for bit in binary_sequence])
    s_obs = abs(s - n/2) / np.sqrt(n/4)

    # Calculate p-value
    p_value = math.erfc(s_obs / np.sqrt(2))
    return p_value

# Runs Test
def runs_test(binary_sequence):
    """
    Analyzes the number of runs (consecutive sequences of the same bit)

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    n = len(binary_sequence)
    pi = np.mean([int(bit) for bit in binary_sequence])

    # Calculate the number of runs
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1

    # Calculate test statistic
    runs_obs = runs
    runs_exp = ((2 * n * pi * (1 - pi)) + 1)
    runs_std = 2 * np.sqrt(2 * n * pi * (1 - pi)) - (1/2)

    z = (runs_obs - runs_exp) / runs_std

    # Calculate p-value
    p_value = math.erfc(abs(z) / np.sqrt(2))
    return p_value

# Longest Run of Ones Test
def longest_run_test(binary_sequence):
    """
    Checks the longest run of consecutive 1s in the sequence

    Args:
        binary_sequence (str): Binary sequence of 0s and 1s

    Returns:
        float: p-value indicating randomness
    """
    # Split sequence into blocks
    k = 6  # Number of blocks
    m = 8  # Length of each block

    blocks = [binary_sequence[i:i+m] for i in range(0, len(binary_sequence), m)]

    # Count longest runs in each block
    longest_runs = []
    for block in blocks:
        max_run = 0
        current_run = 0

        # Find the longest run of 1s in the block
        for bit in block:
            if bit == '1':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        longest_runs.append(max_run)

    # Predefined chi-square distribution parameters
    v_obs = [0, 0, 0, 0]
    for run in longest_runs:
        if run <= 1:
            v_obs[0] += 1
        elif run == 2:
            v_obs[1] += 1
        elif run == 3:
            v_obs[2] += 1
        elif run >= 4:
            v_obs[3] += 1

    # Predefined probabilities for chi-square
    pi = [0.2148, 0.3672, 0.2305, 0.1875]

    # Chi-square calculation
    chi_sq = sum(((v_obs[i] - k * pi[i])**2) / (k * pi[i]) for i in range(4))

    # Calculate p-value
    p_value = special.gammainc(2.5, chi_sq/2)
    return p_value

# Function to calculate cryptographic features
def extract_features(hex_data):
    binary_data = hex_to_binary(hex_data)

    # Perform NIST randomness tests
    freq_p_value = frequency_test(binary_data)
    runs_p_value = runs_test(binary_data)
    longest_run_p_value = longest_run_test(binary_data)



    # Combine all the extracted features into a dictionary
    features = {
        'nist_frequency_test_p_value': freq_p_value,
        'nist_runs_test_p_value': runs_p_value,
        'nist_longest_run_test_p_value': longest_run_p_value,

    }

    return features

# Assuming 'Encrypted Data (Hex)' column exists in the DataFrame
nist_p_values_list = []
nist_p_values_runs_test_list = []
nist_p_values_longest_run_list = []

for i in df['Encrypted Data (Hex)']:
    features = extract_features(i)

    nist_p_values_list.append(features['nist_frequency_test_p_value'])
    nist_p_values_runs_test_list.append(features['nist_runs_test_p_value'])
    nist_p_values_longest_run_list.append(features['nist_longest_run_test_p_value'])


# Add the results to the dataframe
df['nist_frequency_test_p_value'] = nist_p_values_list
df['nist_runs_test_p_value'] = nist_p_values_runs_test_list
df['nist_longest_run_test_p_value'] = nist_p_values_longest_run_list

"""#Preparing Testing Dataset"""

y_test = df['Algorithm']
df.drop(columns=['Original Text', 'Length', 'Encrypted Data (Binary)', 'Encrypted Data (Hex)', 'Algorithm', 'iv'], inplace=True)
X_test = df

y_test

X_test

"""##**Data Preprocessing and Feature Engineering (from Object Columns) for Training Dataset**"""

y_train.isnull().sum()

for i in X_train.isnull().sum():
    if i != 0:
        print('Null value exists')

for i in X_train:
    if X_train[i].dtype == 'object':
        print(i)

X_train['mode']

# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_train['block_size'] = imputer.fit_transform(X_train[['block_size']])
# X_train.isnull().sum()

from sklearn.preprocessing import LabelEncoder

label_encoder1 = LabelEncoder()
label_encoder1.fit(X_train[['mode']])
X_train['mode'] = label_encoder1.transform(X_train[['mode']])
X_train['mode']

X_train['byte_value_histogram']

import numpy as np
from scipy.stats import entropy
from scipy.stats import skew, kurtosis
byte_value_histogram_mean_list = []
byte_value_histogram_std_dev_list = []
byte_distribution_entropy_list = []
byte_distribution_uniformity_score_list = []
byte_distribution_peak_to_mean_ratio_list = []
byte_distribution_low_frequency_byte_count_list = []
byte_distribution_skewness_list = []
byte_distribution_kurtosis_list = []
byte_distribution_dominant_byte_frequency_list = []
byte_distribution_byte_range_spread_list = []
for i in X_train['byte_value_histogram']:
    l = i.split(', ')
    l1 = []
    for j in l:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        l1.append(int(j))
    byte_value_histogram_mean_list.append(sum(l1)/len(l1))
    byte_value_histogram_std_dev_list.append(np.std(l1))
    total_sum = sum(l1)
    byte_distribution = [i / total_sum for i in l1]
    byte_distribution_entropy_list.append(entropy(byte_distribution, base=2))
    ideal_uniform = 1 / 256
    byte_distribution_uniformity_score_list.append(1 - np.sum(np.abs(np.array(byte_distribution) - ideal_uniform)) / 2)
    mean_frequency = np.mean(byte_distribution)
    peak_frequency = max(byte_distribution)
    byte_distribution_peak_to_mean_ratio_list.append(peak_frequency / mean_frequency)
    byte_distribution_low_frequency_byte_count_list.append(sum(1 for freq in byte_distribution if freq < 0.001))
    byte_distribution_skewness_list.append(skew(byte_distribution))
    byte_distribution_kurtosis_list.append(kurtosis(byte_distribution))
    byte_distribution_dominant_byte_frequency_list.append(max(byte_distribution))
    byte_distribution_byte_range_spread_list.append(max(byte_distribution) - min(byte_distribution))
byte_value_histogram_mean_df = pd.DataFrame(byte_value_histogram_mean_list, columns=['byte_value_histogram_mean'])
byte_value_histogram_std_dev_df = pd.DataFrame(byte_value_histogram_std_dev_list, columns=['byte_value_histogram_std_dev'])
byte_distribution_entropy_df = pd.DataFrame(byte_distribution_entropy_list, columns=['byte_distribution_entropy'])
byte_distribution_uniformity_score_df = pd.DataFrame(byte_distribution_uniformity_score_list, columns=['byte_distribution_uniformity_score'])
byte_distribution_peak_to_mean_ratio_df = pd.DataFrame(byte_distribution_peak_to_mean_ratio_list, columns=['byte_distribution_peak_to_mean_ratio'])
byte_distribution_low_frequency_byte_count_df = pd.DataFrame(byte_distribution_low_frequency_byte_count_list, columns=['byte_distribution_low_frequency_byte_count'])
byte_distribution_skewness_df = pd.DataFrame(byte_distribution_skewness_list, columns=['byte_distribution_skewness'])
byte_distribution_kurtosis_df = pd.DataFrame(byte_distribution_kurtosis_list, columns=['byte_distribution_kurtosis'])
byte_distribution_dominant_byte_frequency_df = pd.DataFrame(byte_distribution_dominant_byte_frequency_list, columns=['byte_distribution_dominant_byte_frequency'])
byte_distribution_byte_range_spread_df = pd.DataFrame(byte_distribution_byte_range_spread_list, columns=['byte_distribution_byte_range_spread'])
X_train = pd.concat([X_train, byte_value_histogram_mean_df, byte_value_histogram_std_dev_df, byte_distribution_entropy_df, byte_distribution_uniformity_score_df, byte_distribution_peak_to_mean_ratio_df, byte_distribution_low_frequency_byte_count_df, byte_distribution_skewness_df, byte_distribution_kurtosis_df, byte_distribution_dominant_byte_frequency_df, byte_distribution_byte_range_spread_df], axis=1)
X_train

X_train['byte_value_histogram_mean']

X_train.drop(columns=['byte_value_histogram'], inplace=True)

X_train['byte_value_percentiles']

import numpy as np
byte_value_25th_percentile_list = []
byte_value_50th_percentile_list = []
byte_value_75th_percentile_list = []
for i in X_train['byte_value_percentiles']:
    l = i.split(', ')
    l1 = []
    for j in l:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        l1.append(float(j))
    byte_value_25th_percentile_list.append(l1[0])
    byte_value_50th_percentile_list.append(l1[1])
    byte_value_75th_percentile_list.append(l1[2])
byte_value_25th_percentile_df = pd.DataFrame(byte_value_25th_percentile_list, columns=['byte_value_25th_percentile'])
byte_value_50th_percentile_df = pd.DataFrame(byte_value_50th_percentile_list, columns=['byte_value_50th_percentile'])
byte_value_75th_percentile_df = pd.DataFrame(byte_value_75th_percentile_list, columns=['byte_value_75th_percentile'])
X_train = pd.concat([X_train, byte_value_25th_percentile_df, byte_value_50th_percentile_df, byte_value_75th_percentile_df], axis=1)
X_train

X_train['byte_value_50th_percentile']

X_train.drop(columns='byte_value_percentiles', inplace=True)

for i in X_train:
    if X_train[i].dtype == 'object':
        print(i)

X_train['freq_byte_value_diff']

import numpy as np
freq_byte_value_diff_mean_keys_list = []
freq_byte_value_diff_mean_values_list = []
freq_byte_value_diff_weighted_mean_list = []
freq_byte_value_diff_max_keys_list = []
freq_byte_value_diff_max_values_list = []
freq_byte_value_diff_std_dev_keys_list = []
for i in X_train['freq_byte_value_diff']:
    l = i.split(', ')
    d = {}
    for j in l:
        if (j[0] == '{'):
            j = j[1:]
        if (j[-1] == '}'):
            j = j[:-1]
        l1 = j.split(': ')
        d[int(l1[0])] = int(l1[1])
    freq_byte_value_diff_mean_keys_list.append(np.mean(list(d.keys())))
    freq_byte_value_diff_mean_values_list.append(np.mean(list(d.values())))
    freq_byte_value_diff_weighted_mean_list.append(np.average(list(d.keys()), weights=list(d.values())))
    freq_byte_value_diff_max_keys_list.append(max(list(d.keys())))
    freq_byte_value_diff_max_values_list.append(max(list(d.values())))
    freq_byte_value_diff_std_dev_keys_list.append(np.std(list(d.keys())))
freq_byte_value_diff_mean_keys_df = pd.DataFrame(freq_byte_value_diff_mean_keys_list, columns=['freq_byte_value_diff_mean_keys'])
freq_byte_value_diff_mean_values_df = pd.DataFrame(freq_byte_value_diff_mean_values_list, columns=['freq_byte_value_diff_mean_values'])
freq_byte_value_diff_weighted_mean_df = pd.DataFrame(freq_byte_value_diff_weighted_mean_list, columns=['freq_byte_value_diff_weighted_mean'])
freq_byte_value_diff_max_keys_df = pd.DataFrame(freq_byte_value_diff_max_keys_list, columns=['freq_byte_value_diff_max_keys'])
freq_byte_value_diff_max_values_df = pd.DataFrame(freq_byte_value_diff_max_values_list, columns=['freq_byte_value_diff_max_values'])
freq_byte_value_diff_std_dev_keys_df = pd.DataFrame(freq_byte_value_diff_std_dev_keys_list, columns=['freq_byte_value_diff_std_dev_keys'])
X_train = pd.concat([X_train, freq_byte_value_diff_mean_keys_df, freq_byte_value_diff_mean_values_df, freq_byte_value_diff_weighted_mean_df, freq_byte_value_diff_max_keys_df, freq_byte_value_diff_max_values_df, freq_byte_value_diff_std_dev_keys_df], axis=1)
X_train

X_train['freq_byte_value_diff_max_keys']

X_train.drop(columns=['freq_byte_value_diff'], inplace=True)

X_train['run_length_encoding']

import numpy as np
run_length_encoding_total_encoding_length_list = []
run_length_encoding_max_run_length_list = []
run_length_encoding_mean_run_length_list = []
run_length_encoding_std_dev_run_length_list = []
run_length_encoding_mean_value_list = []
run_length_encoding_std_dev_value_list = []
for i in X_train['run_length_encoding']:
    l = i.split(', ')
    run = []
    value = []
    parity = 0
    for j in l:
        if (j == ''):
            run.append(0)
            value.append(0)
            continue
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if (parity == 0):
            j = j[1:]
            if ((len(j) > 1) and (j[-1] == ',')):
                j = j[:-1]
            if (j == ''):
                run.append(0)
            else:
                run.append(int(j))
            parity = 1
        else:
            j = j[:-1]
            if ((len(j) > 1) and (j[-1] == ',')):
                j = j[:-1]
            if (j == ''):
                value.append(0)
            else:
                if (j[-1] == ')'):
                    j = j[:-1]
                value.append(int(j))
            parity = 0
    run_length_encoding_total_encoding_length_list.append(sum(run))
    run_length_encoding_max_run_length_list.append(max(run))
    run_length_encoding_mean_run_length_list.append(np.mean(run))
    run_length_encoding_std_dev_run_length_list.append(np.std(run))
    run_length_encoding_mean_value_list.append(np.mean(value))
    run_length_encoding_std_dev_value_list.append(np.std(value))
run_length_encoding_total_encoding_length_df = pd.DataFrame(run_length_encoding_total_encoding_length_list, columns=['run_length_encoding_total_encoding_length'])
run_length_encoding_max_run_length_df = pd.DataFrame(run_length_encoding_max_run_length_list, columns=['run_length_encoding_max_run_length'])
run_length_encoding_mean_run_length_df = pd.DataFrame(run_length_encoding_mean_run_length_list, columns=['run_length_encoding_mean_run_length'])
run_length_encoding_std_dev_run_length_df = pd.DataFrame(run_length_encoding_std_dev_run_length_list, columns=['run_length_encoding_std_dev_run_length'])
run_length_encoding_mean_value_df = pd.DataFrame(run_length_encoding_mean_value_list, columns=['run_length_encoding_mean_value'])
run_length_encoding_std_dev_value_df = pd.DataFrame(run_length_encoding_std_dev_value_list, columns=['run_length_encoding_std_dev_value'])
X_train = pd.concat([X_train, run_length_encoding_total_encoding_length_df, run_length_encoding_max_run_length_df, run_length_encoding_mean_run_length_df, run_length_encoding_std_dev_run_length_df, run_length_encoding_mean_value_df, run_length_encoding_std_dev_value_df], axis=1)
X_train

X_train['run_length_encoding_mean_run_length']

X_train.drop(columns=['run_length_encoding'], inplace=True)

X_train['byte_value_transition_matrix']

from scipy.stats import entropy
import numpy as np
byte_value_transition_matrix_sparsity_list = []
byte_value_transition_matrix_entropy_list = []
byte_value_transition_matrix_top_k_sum_list = []
byte_value_transition_matrix_mean_prob_per_row_list = []
byte_value_transition_matrix_quadrant_1_sum_list = []
byte_value_transition_matrix_quadrant_2_sum_list = []
byte_value_transition_matrix_quadrant_3_sum_list = []
byte_value_transition_matrix_quadrant_4_sum_list = []
for i in X_train['byte_value_transition_matrix']:
    l = []
    l1 = i.split(', ')
    l2 = []
    for j in l1:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if ((len(j) > 1) and (j[-1] == ',')):
            j = j[:-1]
        if (type(j) == list):
            l.append(j)
        else:
            l2.append(int(j))
        if (len(l2) == 256):
            l1.append(l2)
            l2 = []
    l = np.array(l)
    sparsity = 1 - np.count_nonzero(l) / (256 * 256)
    byte_value_transition_matrix_sparsity_list.append(sparsity)
    prob_matrix = l / np.sum(l)
    byte_value_transition_matrix_entropy_list.append(entropy(prob_matrix.flatten(), base=2))
    top_k_transitions = np.sort(l.flatten())[::-1][:5]  # Sum of top 5
    top_k_sum = np.sum(top_k_transitions)
    byte_value_transition_matrix_top_k_sum_list.append(top_k_sum)
    normalized_matrix = l / (np.sum(l, axis=1, keepdims=True) + 1e-10)
    byte_value_transition_matrix_mean_prob_per_row_list.append(np.mean(normalized_matrix, axis=1).mean())
    byte_value_transition_matrix_quadrant_1_sum_list.append(np.sum(l[:128, :128]))  # Top-left
    byte_value_transition_matrix_quadrant_2_sum_list.append(np.sum(l[:128, 128:]))  # Top-right
    byte_value_transition_matrix_quadrant_3_sum_list.append(np.sum(l[128:, :128]))  # Bottom-left
    byte_value_transition_matrix_quadrant_4_sum_list.append(np.sum(l[128:, 128:]))  # Bottom-right
byte_value_transition_matrix_sparsity_df = pd.DataFrame(byte_value_transition_matrix_sparsity_list, columns=['byte_value_transition_matrix_sparsity'])
byte_value_transition_matrix_entropy_df = pd.DataFrame(byte_value_transition_matrix_entropy_list, columns=['byte_value_transition_matrix_entropy'])
byte_value_transition_matrix_top_k_sum_df = pd.DataFrame(byte_value_transition_matrix_top_k_sum_list, columns=['byte_value_transition_matrix_top_k_sum'])
byte_value_transition_matrix_mean_prob_per_row_df = pd.DataFrame(byte_value_transition_matrix_mean_prob_per_row_list, columns=['byte_value_transition_matrix_mean_prob_per_row'])
byte_value_transition_matrix_quadrant_1_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_1_sum_list, columns=['byte_value_transition_matrix_quadrant_1_sum'])
byte_value_transition_matrix_quadrant_2_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_2_sum_list, columns=['byte_value_transition_matrix_quadrant_2_sum'])
byte_value_transition_matrix_quadrant_3_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_3_sum_list, columns=['byte_value_transition_matrix_quadrant_3_sum'])
byte_value_transition_matrix_quadrant_4_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_4_sum_list, columns=['byte_value_transition_matrix_quadrant_4_sum'])
X_train = pd.concat([X_train, byte_value_transition_matrix_sparsity_df, byte_value_transition_matrix_entropy_df, byte_value_transition_matrix_top_k_sum_df, byte_value_transition_matrix_mean_prob_per_row_df, byte_value_transition_matrix_quadrant_1_sum_df, byte_value_transition_matrix_quadrant_2_sum_df, byte_value_transition_matrix_quadrant_3_sum_df, byte_value_transition_matrix_quadrant_4_sum_df], axis=1)
X_train

X_train['byte_value_transition_matrix_mean_prob_per_row']

X_train.drop(columns=['byte_value_transition_matrix'], inplace=True)

X_train['freq_byte_value_2grams']

# X_train.drop(columns=['byte_value_transition_matrix'], inplace=True)

X_train.drop(columns=['freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams'], inplace=True)

X_train['byte_value_acf']

import numpy as np
import pandas as pd
import ast

# Function to summarize ACF for a single ciphertext
def summarize_acf(acf_values):
    # Ensure the input is a list of floats
    acf_values = np.array(acf_values, dtype=np.float64)

    # Exclude lag 0 (self-correlation)
    acf_no_lag0 = acf_values[1:]

    # Compute features
    mean_acf = np.mean(acf_no_lag0)
    variance_acf = np.var(acf_no_lag0)
    max_acf = np.max(acf_no_lag0)
    lag_of_max_acf = np.argmax(acf_no_lag0) + 1  # +1 because we excluded lag 0

    return {
        "mean_acf": mean_acf,
        "variance_acf": variance_acf,
        "max_acf": max_acf,
        "lag_of_max_acf": lag_of_max_acf
    }



# Convert the string representation of a list into an actual list
X_train['byte_value_acf'] = X_train['byte_value_acf'].apply(lambda x: ast.literal_eval(x))

# Apply summarize_acf for each row in 'byte_value_acf' column
features_list = X_train['byte_value_acf'].apply(summarize_acf)

# Convert features_list into a DataFrame
features_df = pd.DataFrame(features_list.tolist())

# Add the new features to the original X_train
X_train = pd.concat([X_train, features_df], axis=1)

# Print the updated X_train with extracted features
X_train

X_train['max_acf']

X_train.drop(columns=['byte_value_acf'], inplace=True)

X_train['byte_value_power_spectrum']

import numpy as np
import pandas as pd

# Function to calculate total power (sum of all frequencies)
def total_power(power_spectrum):
    return np.sum(power_spectrum)

# Function to calculate peak power (maximum frequency component)
def peak_power(power_spectrum):
    return np.max(power_spectrum)

# Function to calculate power concentration (ratio of top n frequencies' power to total power)
def power_concentration(power_spectrum, top_n=3):
    sorted_spectrum = np.sort(power_spectrum)[::-1]  # Sort in descending order
    return np.sum(sorted_spectrum[:top_n]) / np.sum(power_spectrum)

# # Example of applying these functions to your data
# X_train = pd.DataFrame({
#     'byte_value_power_spectrum': [
#         [150100455184.0, 10933011.937147308, 17569885.488023052],
#         [60031504.0, 400436.77105714486, 428221.7924971855],
#         [11758041.0, 98795.90679018006, 258054.280108135],
#         [3150625.0, 36711.263661699806, 240051.794852449],
#         [619369.0, 18911.38498657157, 91709.0000000000]
#     ]
# })

l = []
for i in X_train['byte_value_power_spectrum']:
    l1 = []
    l2 = i.split(", ")
    for j in l2:
      if (j == ''):
          l1.append(0)
          continue
      if (j[0] == '['):
          j = j[1:]
      if (j[-1] == ']'):
          j = j[:-1]
      if (j[-1] == ','):
          j = j[:-1]
      l1.append(float(j))
    l.append(l1)
X_train['byte_value_power_spectrum'] = l

# Apply the functions to each row in the 'byte_value_power_spectrum' column
X_train['total_power'] = X_train['byte_value_power_spectrum'].apply(lambda x: total_power(x))
X_train['peak_power'] = X_train['byte_value_power_spectrum'].apply(lambda x: peak_power(x))
X_train['power_concentration'] = X_train['byte_value_power_spectrum'].apply(lambda x: power_concentration(x))

# Display the resulting DataFrame with new features
X_train

X_train['byte_value_power_spectrum']

X_train['power_concentration']

X_train['peak_power']

X_train['total_power']

X_train.drop(columns=['byte_value_power_spectrum'], inplace=True)

X_train

"""#Feature List

"""

X_train.columns
features=list(X_train.columns)
features

"""#Imputing Null Values in Training Dataset"""

for i in X_train:
    if (X_train[i].isnull().sum() != 0):
        print(i)

import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_train['byte_value_transition_matrix_entropy'] = imputer.fit_transform(X_train[['byte_value_transition_matrix_entropy']])
X_train['byte_value_transition_matrix_entropy'].isnull().sum()

X_train

y_train

"""#Extracting Training Dataset for CNN"""

X_train_cnn = X_train
y_train_cnn = y_train

"""#Normalization of Training Dataset"""

from sklearn.preprocessing import StandardScaler
exclude_columns = ['mode', 'block_size', 'block_cipher_boolean', 'block_frequency', 'length', 'byte_distribution_uniformity_score', 'byte_distribution_low_frequency_byte_count', 'byte_distribution_skewness', 'byte_distribution_kurtosis', 'byte_distribution_dominant_byte_frequency', 'byte_distribution_byte_range_spread']
columns_to_scale = X_train.columns.difference(exclude_columns)

scaler = StandardScaler()

scaler.fit(X_train[columns_to_scale])
X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
X_train

"""##**Data Preprocessing and Feature Engineering (from Object Columns) for Testing Dataset**"""

y_test.isnull().sum()

X_test.isnull().sum()

for i in X_test:
    if X_test[i].dtype == 'object':
        print(i)

# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_test['block_size'] = imputer.fit_transform(X_test[['block_size']])
# X_test.isnull().sum()

X_test['mode'] = label_encoder1.transform(X_test[['mode']])
X_test

X_test['byte_value_histogram']

import numpy as np
from scipy.stats import entropy
from scipy.stats import skew, kurtosis
byte_value_histogram_mean_list = []
byte_value_histogram_std_dev_list = []
byte_distribution_entropy_list = []
byte_distribution_uniformity_score_list = []
byte_distribution_peak_to_mean_ratio_list = []
byte_distribution_low_frequency_byte_count_list = []
byte_distribution_skewness_list = []
byte_distribution_kurtosis_list = []
byte_distribution_dominant_byte_frequency_list = []
byte_distribution_byte_range_spread_list = []
for i in X_test['byte_value_histogram']:
    l = i.split(', ')
    l1 = []
    for j in l:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        l1.append(int(j))
    byte_value_histogram_mean_list.append(sum(l1)/len(l1))
    byte_value_histogram_std_dev_list.append(np.std(l1))
    total_sum = sum(l1)
    byte_distribution = [i / total_sum for i in l1]
    byte_distribution_entropy_list.append(entropy(byte_distribution, base=2))
    ideal_uniform = 1 / 256
    byte_distribution_uniformity_score_list.append(1 - np.sum(np.abs(np.array(byte_distribution) - ideal_uniform)) / 2)
    mean_frequency = np.mean(byte_distribution)
    peak_frequency = max(byte_distribution)
    byte_distribution_peak_to_mean_ratio_list.append(peak_frequency / mean_frequency)
    byte_distribution_low_frequency_byte_count_list.append(sum(1 for freq in byte_distribution if freq < 0.001))
    byte_distribution_skewness_list.append(skew(byte_distribution))
    byte_distribution_kurtosis_list.append(kurtosis(byte_distribution))
    byte_distribution_dominant_byte_frequency_list.append(max(byte_distribution))
    byte_distribution_byte_range_spread_list.append(max(byte_distribution) - min(byte_distribution))
byte_value_histogram_mean_df = pd.DataFrame(byte_value_histogram_mean_list, columns=['byte_value_histogram_mean'])
byte_value_histogram_std_dev_df = pd.DataFrame(byte_value_histogram_std_dev_list, columns=['byte_value_histogram_std_dev'])
byte_distribution_entropy_df = pd.DataFrame(byte_distribution_entropy_list, columns=['byte_distribution_entropy'])
byte_distribution_uniformity_score_df = pd.DataFrame(byte_distribution_uniformity_score_list, columns=['byte_distribution_uniformity_score'])
byte_distribution_peak_to_mean_ratio_df = pd.DataFrame(byte_distribution_peak_to_mean_ratio_list, columns=['byte_distribution_peak_to_mean_ratio'])
byte_distribution_low_frequency_byte_count_df = pd.DataFrame(byte_distribution_low_frequency_byte_count_list, columns=['byte_distribution_low_frequency_byte_count'])
byte_distribution_skewness_df = pd.DataFrame(byte_distribution_skewness_list, columns=['byte_distribution_skewness'])
byte_distribution_kurtosis_df = pd.DataFrame(byte_distribution_kurtosis_list, columns=['byte_distribution_kurtosis'])
byte_distribution_dominant_byte_frequency_df = pd.DataFrame(byte_distribution_dominant_byte_frequency_list, columns=['byte_distribution_dominant_byte_frequency'])
byte_distribution_byte_range_spread_df = pd.DataFrame(byte_distribution_byte_range_spread_list, columns=['byte_distribution_byte_range_spread'])
X_test = pd.concat([X_test, byte_value_histogram_mean_df, byte_value_histogram_std_dev_df, byte_distribution_entropy_df, byte_distribution_uniformity_score_df, byte_distribution_peak_to_mean_ratio_df, byte_distribution_low_frequency_byte_count_df, byte_distribution_skewness_df, byte_distribution_kurtosis_df, byte_distribution_dominant_byte_frequency_df, byte_distribution_byte_range_spread_df], axis=1)
X_test

X_test['byte_value_histogram_mean']

X_test.drop(columns=['byte_value_histogram'], inplace=True)

X_test['byte_value_percentiles']

import numpy as np
byte_value_25th_percentile_list = []
byte_value_50th_percentile_list = []
byte_value_75th_percentile_list = []
for i in X_test['byte_value_percentiles']:
    l = i.split(', ')
    l1 = []
    for j in l:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        l1.append(float(j))
    byte_value_25th_percentile_list.append(l1[0])
    byte_value_50th_percentile_list.append(l1[1])
    byte_value_75th_percentile_list.append(l1[2])
byte_value_25th_percentile_df = pd.DataFrame(byte_value_25th_percentile_list, columns=['byte_value_25th_percentile'])
byte_value_50th_percentile_df = pd.DataFrame(byte_value_50th_percentile_list, columns=['byte_value_50th_percentile'])
byte_value_75th_percentile_df = pd.DataFrame(byte_value_75th_percentile_list, columns=['byte_value_75th_percentile'])
X_test = pd.concat([X_test, byte_value_25th_percentile_df, byte_value_50th_percentile_df, byte_value_75th_percentile_df], axis=1)
X_test

X_test['byte_value_50th_percentile']

X_test.drop(columns='byte_value_percentiles', inplace=True)

for i in X_test:
    if X_test[i].dtype == 'object':
        print(i)

X_test['freq_byte_value_diff']

import numpy as np
freq_byte_value_diff_mean_keys_list = []
freq_byte_value_diff_mean_values_list = []
freq_byte_value_diff_weighted_mean_list = []
freq_byte_value_diff_max_keys_list = []
freq_byte_value_diff_max_values_list = []
freq_byte_value_diff_std_dev_keys_list = []
for i in X_test['freq_byte_value_diff']:
    l = i.split(', ')
    d = {}
    for j in l:
        if (j[0] == '{'):
            j = j[1:]
        if (j[-1] == '}'):
            j = j[:-1]
        l1 = j.split(': ')
        d[int(l1[0])] = int(l1[1])
    freq_byte_value_diff_mean_keys_list.append(np.mean(list(d.keys())))
    freq_byte_value_diff_mean_values_list.append(np.mean(list(d.values())))
    freq_byte_value_diff_weighted_mean_list.append(np.average(list(d.keys()), weights=list(d.values())))
    freq_byte_value_diff_max_keys_list.append(max(list(d.keys())))
    freq_byte_value_diff_max_values_list.append(max(list(d.values())))
    freq_byte_value_diff_std_dev_keys_list.append(np.std(list(d.keys())))
freq_byte_value_diff_mean_keys_df = pd.DataFrame(freq_byte_value_diff_mean_keys_list, columns=['freq_byte_value_diff_mean_keys'])
freq_byte_value_diff_mean_values_df = pd.DataFrame(freq_byte_value_diff_mean_values_list, columns=['freq_byte_value_diff_mean_values'])
freq_byte_value_diff_weighted_mean_df = pd.DataFrame(freq_byte_value_diff_weighted_mean_list, columns=['freq_byte_value_diff_weighted_mean'])
freq_byte_value_diff_max_keys_df = pd.DataFrame(freq_byte_value_diff_max_keys_list, columns=['freq_byte_value_diff_max_keys'])
freq_byte_value_diff_max_values_df = pd.DataFrame(freq_byte_value_diff_max_values_list, columns=['freq_byte_value_diff_max_values'])
freq_byte_value_diff_std_dev_keys_df = pd.DataFrame(freq_byte_value_diff_std_dev_keys_list, columns=['freq_byte_value_diff_std_dev_keys'])
X_test = pd.concat([X_test, freq_byte_value_diff_mean_keys_df, freq_byte_value_diff_mean_values_df, freq_byte_value_diff_weighted_mean_df, freq_byte_value_diff_max_keys_df, freq_byte_value_diff_max_values_df, freq_byte_value_diff_std_dev_keys_df], axis=1)
X_test

X_test['freq_byte_value_diff_max_keys']

X_test.drop(columns=['freq_byte_value_diff'], inplace=True)

X_test['run_length_encoding']

import numpy as np
run_length_encoding_total_encoding_length_list = []
run_length_encoding_max_run_length_list = []
run_length_encoding_mean_run_length_list = []
run_length_encoding_std_dev_run_length_list = []
run_length_encoding_mean_value_list = []
run_length_encoding_std_dev_value_list = []
for i in X_test['run_length_encoding']:
    l = i.split(', ')
    run = []
    value = []
    parity = 0
    for j in l:
        if (j == ''):
            run.append(0)
            value.append(0)
            continue
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if (parity == 0):
            j = j[1:]
            if ((len(j) > 1) and (j[-1] == ',')):
                j = j[:-1]
            if (j == ''):
                run.append(0)
            else:
                run.append(int(j))
            parity = 1
        else:
            j = j[:-1]
            if ((len(j) > 1) and (j[-1] == ',')):
                j = j[:-1]
            if (j == ''):
                value.append(0)
            else:
                if (j[-1] == ')'):
                    j = j[:-1]
                value.append(int(j))
            parity = 0
    run_length_encoding_total_encoding_length_list.append(sum(run))
    run_length_encoding_max_run_length_list.append(max(run))
    run_length_encoding_mean_run_length_list.append(np.mean(run))
    run_length_encoding_std_dev_run_length_list.append(np.std(run))
    run_length_encoding_mean_value_list.append(np.mean(value))
    run_length_encoding_std_dev_value_list.append(np.std(value))
run_length_encoding_total_encoding_length_df = pd.DataFrame(run_length_encoding_total_encoding_length_list, columns=['run_length_encoding_total_encoding_length'])
run_length_encoding_max_run_length_df = pd.DataFrame(run_length_encoding_max_run_length_list, columns=['run_length_encoding_max_run_length'])
run_length_encoding_mean_run_length_df = pd.DataFrame(run_length_encoding_mean_run_length_list, columns=['run_length_encoding_mean_run_length'])
run_length_encoding_std_dev_run_length_df = pd.DataFrame(run_length_encoding_std_dev_run_length_list, columns=['run_length_encoding_std_dev_run_length'])
run_length_encoding_mean_value_df = pd.DataFrame(run_length_encoding_mean_value_list, columns=['run_length_encoding_mean_value'])
run_length_encoding_std_dev_value_df = pd.DataFrame(run_length_encoding_std_dev_value_list, columns=['run_length_encoding_std_dev_value'])
X_test = pd.concat([X_test, run_length_encoding_total_encoding_length_df, run_length_encoding_max_run_length_df, run_length_encoding_mean_run_length_df, run_length_encoding_std_dev_run_length_df, run_length_encoding_mean_value_df, run_length_encoding_std_dev_value_df], axis=1)
X_test

X_test['run_length_encoding_mean_run_length']

X_test.drop(columns=['run_length_encoding'], inplace=True)

X_test['byte_value_transition_matrix']

from scipy.stats import entropy
import numpy as np
byte_value_transition_matrix_sparsity_list = []
byte_value_transition_matrix_entropy_list = []
byte_value_transition_matrix_top_k_sum_list = []
byte_value_transition_matrix_mean_prob_per_row_list = []
byte_value_transition_matrix_quadrant_1_sum_list = []
byte_value_transition_matrix_quadrant_2_sum_list = []
byte_value_transition_matrix_quadrant_3_sum_list = []
byte_value_transition_matrix_quadrant_4_sum_list = []
for i in X_test['byte_value_transition_matrix']:
    l = []
    l1 = i.split(', ')
    l2 = []
    for j in l1:
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if (j[0] == '['):
            j = j[1:]
        if (j[-1] == ']'):
            j = j[:-1]
        if ((len(j) > 1) and (j[-1] == ',')):
            j = j[:-1]
        if (type(j) == list):
            l.append(j)
        else:
            l2.append(int(j))
        if (len(l2) == 256):
            l1.append(l2)
            l2 = []
    l = np.array(l)
    sparsity = 1 - np.count_nonzero(l) / (256 * 256)
    byte_value_transition_matrix_sparsity_list.append(sparsity)
    prob_matrix = l / np.sum(l)
    byte_value_transition_matrix_entropy_list.append(entropy(prob_matrix.flatten(), base=2))
    top_k_transitions = np.sort(l.flatten())[::-1][:5]  # Sum of top 5
    top_k_sum = np.sum(top_k_transitions)
    byte_value_transition_matrix_top_k_sum_list.append(top_k_sum)
    normalized_matrix = l / (np.sum(l, axis=1, keepdims=True) + 1e-10)
    byte_value_transition_matrix_mean_prob_per_row_list.append(np.mean(normalized_matrix, axis=1).mean())
    byte_value_transition_matrix_quadrant_1_sum_list.append(np.sum(l[:128, :128]))  # Top-left
    byte_value_transition_matrix_quadrant_2_sum_list.append(np.sum(l[:128, 128:]))  # Top-right
    byte_value_transition_matrix_quadrant_3_sum_list.append(np.sum(l[128:, :128]))  # Bottom-left
    byte_value_transition_matrix_quadrant_4_sum_list.append(np.sum(l[128:, 128:]))  # Bottom-right
byte_value_transition_matrix_sparsity_df = pd.DataFrame(byte_value_transition_matrix_sparsity_list, columns=['byte_value_transition_matrix_sparsity'])
byte_value_transition_matrix_entropy_df = pd.DataFrame(byte_value_transition_matrix_entropy_list, columns=['byte_value_transition_matrix_entropy'])
byte_value_transition_matrix_top_k_sum_df = pd.DataFrame(byte_value_transition_matrix_top_k_sum_list, columns=['byte_value_transition_matrix_top_k_sum'])
byte_value_transition_matrix_mean_prob_per_row_df = pd.DataFrame(byte_value_transition_matrix_mean_prob_per_row_list, columns=['byte_value_transition_matrix_mean_prob_per_row'])
byte_value_transition_matrix_quadrant_1_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_1_sum_list, columns=['byte_value_transition_matrix_quadrant_1_sum'])
byte_value_transition_matrix_quadrant_2_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_2_sum_list, columns=['byte_value_transition_matrix_quadrant_2_sum'])
byte_value_transition_matrix_quadrant_3_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_3_sum_list, columns=['byte_value_transition_matrix_quadrant_3_sum'])
byte_value_transition_matrix_quadrant_4_sum_df = pd.DataFrame(byte_value_transition_matrix_quadrant_4_sum_list, columns=['byte_value_transition_matrix_quadrant_4_sum'])
X_test = pd.concat([X_test, byte_value_transition_matrix_sparsity_df, byte_value_transition_matrix_entropy_df, byte_value_transition_matrix_top_k_sum_df, byte_value_transition_matrix_mean_prob_per_row_df, byte_value_transition_matrix_quadrant_1_sum_df, byte_value_transition_matrix_quadrant_2_sum_df, byte_value_transition_matrix_quadrant_3_sum_df, byte_value_transition_matrix_quadrant_4_sum_df], axis=1)
X_test

X_test['byte_value_transition_matrix_mean_prob_per_row']

X_test.drop(columns=['byte_value_transition_matrix'], inplace=True)

X_test['freq_byte_value_2grams']

X_test.drop(columns=['freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams'], inplace=True)

X_test['byte_value_acf']

import numpy as np
import pandas as pd
import ast

# Function to summarize ACF for a single ciphertext
def summarize_acf(acf_values):
    # Ensure the input is a list of floats
    acf_values = np.array(acf_values, dtype=np.float64)

    # Exclude lag 0 (self-correlation)
    acf_no_lag0 = acf_values[1:]

    # Compute features
    mean_acf = np.mean(acf_no_lag0)
    variance_acf = np.var(acf_no_lag0)
    max_acf = np.max(acf_no_lag0)
    lag_of_max_acf = np.argmax(acf_no_lag0) + 1  # +1 because we excluded lag 0

    return {
        "mean_acf": mean_acf,
        "variance_acf": variance_acf,
        "max_acf": max_acf,
        "lag_of_max_acf": lag_of_max_acf
    }



# Convert the string representation of a list into an actual list
X_test['byte_value_acf'] = X_test['byte_value_acf'].apply(lambda x: ast.literal_eval(x))

# Apply summarize_acf for each row in 'byte_value_acf' column
features_list = X_test['byte_value_acf'].apply(summarize_acf)

# Convert features_list into a DataFrame
features_df = pd.DataFrame(features_list.tolist())

# Add the new features to the original X_train
X_test = pd.concat([X_test, features_df], axis=1)

# Print the updated X_train with extracted features
X_test

X_test['max_acf']

X_test.drop(columns=['byte_value_acf'], inplace=True)

X_test['byte_value_power_spectrum']

import numpy as np
import pandas as pd

# Function to calculate total power (sum of all frequencies)
def total_power(power_spectrum):
    return np.sum(power_spectrum)

# Function to calculate peak power (maximum frequency component)
def peak_power(power_spectrum):
    return np.max(power_spectrum)

# Function to calculate power concentration (ratio of top n frequencies' power to total power)
def power_concentration(power_spectrum, top_n=3):
    sorted_spectrum = np.sort(power_spectrum)[::-1]  # Sort in descending order
    return np.sum(sorted_spectrum[:top_n]) / np.sum(power_spectrum)

# # Example of applying these functions to your data
# X_train = pd.DataFrame({
#     'byte_value_power_spectrum': [
#         [150100455184.0, 10933011.937147308, 17569885.488023052],
#         [60031504.0, 400436.77105714486, 428221.7924971855],
#         [11758041.0, 98795.90679018006, 258054.280108135],
#         [3150625.0, 36711.263661699806, 240051.794852449],
#         [619369.0, 18911.38498657157, 91709.0000000000]
#     ]
# })

l = []
for i in X_test['byte_value_power_spectrum']:
    l1 = []
    l2 = i.split(", ")
    for j in l2:
      if (j == ''):
          l1.append(0)
          continue
      if (j[0] == '['):
          j = j[1:]
      if (j[-1] == ']'):
          j = j[:-1]
      if (j[-1] == ','):
          j = j[:-1]
      l1.append(float(j))
    l.append(l1)
X_test['byte_value_power_spectrum'] = l

# Apply the functions to each row in the 'byte_value_power_spectrum' column
X_test['total_power'] = X_test['byte_value_power_spectrum'].apply(lambda x: total_power(x))
X_test['peak_power'] = X_test['byte_value_power_spectrum'].apply(lambda x: peak_power(x))
X_test['power_concentration'] = X_test['byte_value_power_spectrum'].apply(lambda x: power_concentration(x))

# Display the resulting DataFrame with new features
X_test

X_test['byte_value_power_spectrum']

X_test['power_concentration']

X_test['peak_power']

X_test['total_power']

X_test.drop(columns=['byte_value_power_spectrum'], inplace=True)

X_test

# columns = []
# for i in X_test:
#     if X_test[i].dtype == 'object':
#         columns.append(i)
# X_test.drop(columns=columns, inplace=True)
# X_test

"""#Imputing Null Values for Testing Dataset"""

for i in X_test:
    if (X_test[i].isnull().sum() != 0):
        print(i)

X_test['byte_value_transition_matrix_entropy'] = imputer.transform(X_test[['byte_value_transition_matrix_entropy']])
X_test['byte_value_transition_matrix_entropy'].isnull().sum()

X_test['serial_correlation'].isnull().sum()

import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_test['serial_correlation'] = imputer.fit_transform(X_test[['serial_correlation']])
X_test['serial_correlation'].isnull().sum()

X_test

y_test

"""#Extracting Testing Dataset for CNN"""

X_test_cnn = X_test
y_test_cnn = y_test

"""#Normalization of Testing Dataset"""

exclude_columns = ['mode', 'block_size', 'block_cipher_boolean', 'block_frequency', 'length', 'byte_distribution_uniformity_score', 'byte_distribution_low_frequency_byte_count', 'byte_distribution_skewness', 'byte_distribution_kurtosis', 'byte_distribution_dominant_byte_frequency', 'byte_distribution_byte_range_spread']
columns_to_scale = X_test.columns.difference(exclude_columns)
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
X_test

"""##**Calculation of Correlation**"""

import seaborn as sns
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Assuming X_train and y_train are your test data
# Convert y_train to a DataFrame for easier manipulation
y_train_correlation_df = pd.DataFrame(y_train_encoded, columns=['Algorithm'])

# Combine X_train and y_train into a single DataFrame
correlation_df = pd.concat([X_train, y_train_correlation_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = correlation_df.corr()

# Extract the correlation of each feature with y_train
correlation_with_algorithm = correlation_matrix['Algorithm'].drop('Algorithm')

# Print the correlation of each feature with y_train
print(correlation_with_algorithm)

# Plot the correlation of each feature with y_train
plt.figure(figsize=(32, 16))
sns.barplot(x=correlation_with_algorithm.index, y=correlation_with_algorithm.values, palette='coolwarm')
plt.xlabel('Features')
plt.ylabel('Correlation with Algorithm')
plt.title('Correlation of Features in X_train with y_train')
plt.xticks(rotation=90)
plt.show()
plt.savefig("Correlation_Plot.png")

y_train

"""# Pair Plots Between Features, SMOTE and PCA analysis

No relevant insights could be drawn from the Pair Plots between
features.

Since the dataset is already balanced, we haven't implemented SMOTE as it won't add any value to the analysis.

PCA was used but didn't result in increased accuracy, so it was dropped.

#Pair Plots
"""

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Example: df is your DataFrame
# df_train = pd.concat([X_train, y_train], axis=1)

# # # Calculate the correlation matrix
# # correlation_matrix = df_train.corr()
# sns.pairplot(df_train, hue='Algorithm')  # Replace 'target' with your actual target column
# # print(correlation_matrix)
# plt.show()

"""#SMOTE"""

# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

"""#PCA"""

# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)  # Adjust number of components
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

"""#**Model Training**

##**XGBoost**
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install xgboost

# Importing necessary libraries
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



# Initialize the XGBoost model for multi-class classification
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',  # Multi-class classification
    num_class=14,               # Number of classes (14 in your case)
    eval_metric='mlogloss'      # Multi-class log loss metric
)

label_encoder_xgboost = LabelEncoder()
# Fit and transform the target variables to numerical values
y_train_encoded = label_encoder_xgboost.fit_transform(y_train)
y_test_encoded = label_encoder_xgboost.transform(y_test)

# Fit the model on the training data
xgb_model.fit(X_train, y_train_encoded)

# Make predictions on the validation set
y_pred_xgb_encoded = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred_xgb_encoded)
print(f"Accuracy: {accuracy * 100:.2f}%")
y_pred_xgb = label_encoder_xgboost.inverse_transform(y_pred_xgb_encoded)
# Print detailed classification report
print(classification_report(y_test, y_pred_xgb))

"""##Confusion Matrix"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb, labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])

# Display the confusion matrix
plt.figure(figsize=(64, 8))  # Width: 10, Height: 8
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])
disp.plot(cmap=plt.cm.Greens, ax=plt.gca())
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb, labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])

# Normalize the confusion matrix to percentages (row-wise normalization)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Display the confusion matrix with percentages
plt.figure(figsize=(64, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])
disp.plot(cmap="Greens", values_format=".2f", ax=plt.gca())  # Format values to 2 decimal places
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.title("Confusion Matrix (Percentages)")
plt.show()

"""##**Random Forest Model**"""

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
# print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print(classification_report(y_test, y_pred))

"""##Confusion Matrix"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])

# Display the confusion matrix
plt.figure(figsize=(64, 8))  # Width: 10, Height: 8
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])

# Normalize the confusion matrix to percentages (row-wise normalization)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Display the confusion matrix with percentages
plt.figure(figsize=(64, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["AES", "3DES", "Rabbit", "ChaCha20", "RSA", "ECC", "SHA3_512", "MD5", "ECDSA", "HMAC"])
disp.plot(cmap="Blues", values_format=".2f", ax=plt.gca())  # Format values to 2 decimal places
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.title("Confusion Matrix (Percentages)")
plt.show()

"""#**Heirarchial Models**"""

print(rf_model.classes_)
rf_labels = rf_model.classes_
# Define your custom mapping
rf_mapping = {}
for i in range(0, len(rf_labels)):
  rf_mapping[rf_labels[i]] = i

# Encode using the mapping
y_test_encoded_rf = [rf_mapping[i] for i in y_test]
print("Encoded labels:", y_test_encoded_rf)

"""##Finding Top-k Accuracy"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Predict probabilities
probs = rf_model.predict_proba(X_test)
# print(probs)

# Get the true labels
true_labels = np.array(y_test_encoded_rf)

# Function to calculate top-k accuracy
def top_k_accuracy(probs, true_labels, k):
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # Get indices of top-k classes
    # print(top_k_preds)
    correct = sum(true_labels[i] in top_k_preds[i] for i in range(len(true_labels)))
    return correct / len(true_labels)

k_value_list = [1, 2, 3, 4, 5]
accuracy_list = []
# Calculate top-k accuracies
for k in range(1, 6):  # Top-1 to Top-5
    accuracy = top_k_accuracy(probs, true_labels, k)
    print(f"Top-{k} Accuracy: {accuracy:.2f}")
    accuracy_list.append(accuracy)

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(8, 6))
plt.plot(k_value_list, accuracy_list, marker='o', linestyle='-', color='b', label='Accuracy')

# Labels and Title
plt.xlabel('k', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs k', fontsize=14)
plt.xticks(k_value_list)  # Show x-axis ticks only at k values
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder_hierarchial_rf = LabelEncoder()
y_train_encoded_hierarchial_rf = label_encoder_hierarchial_rf.fit_transform(y_train)
y_test_encoded_hierarchial_rf = label_encoder_hierarchial_rf.transform(y_test)

"""##**Heirarchial Model with Random Forest**"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Train the first Random Forest model for top-3 prediction
def train_top3_model(X_train, y_train_encoded_hierarchial_rf):
    model_top3 = RandomForestClassifier(random_state=42)
    model_top3.fit(X_train, y_train_encoded_hierarchial_rf)
    return model_top3

# Step 2: Create and train models for all combinations of top-3 classes
def train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes):
    combination_models = {}
    count = 0
    for combination in combinations(all_classes, 3):
        # Filter the dataset for the current combination
        mask = pd.Series(y_train_encoded_hierarchial_rf).isin(combination)
        X_subset = pd.DataFrame(X_train[mask], columns=X_train.columns)
        # print(X_subset)
        # break
        # print(X_subset)
        y_subset = y_train_encoded_hierarchial_rf[mask]
        count = count + 1
        print(count)
        if (X_subset.shape[0] != 0):
            # Train a RandomForestClassifier for this combination
            model = RandomForestClassifier(random_state=42)
            model.fit(X_subset, y_subset)
            combination_models[combination] = model
    return combination_models

# Step 3: Top-3 prediction and refinement
def predict_with_top3_model(X_test, model_top3, combination_models, all_classes):
    # Step 3.1: Predict top-3 classes using model_top4
    probs = model_top3.predict_proba(X_test)
    top3_predictions = np.argsort(probs, axis=1)[:, -3:]  # Indices of top-3 classes

    final_predictions = []
    for i in range(len(X_test)):
        # Step 3.2: Get the actual top-3 class labels
        top3_classes = [all_classes[j] for j in top3_predictions[i]]
        top3_classes_tuple = tuple(sorted(top3_classes))  # Match the order in combination_models

        # Step 3.3: Use the corresponding combination model
        if top3_classes_tuple in combination_models:
            model = combination_models[top3_classes_tuple]
            final_predictions.append(model.predict(pd.DataFrame([X_test.iloc[i]], columns=X_test.columns))[0])
        else:
            # If no specific model is available, default to the top-3 class with highest probability
            final_predictions.append(top3_classes[-1])  # Most likely class
    return final_predictions

all_classes = sorted(list(set(y_train_encoded_hierarchial_rf)))  # Unique classes (14 in this case)

# Train the top-3 model
model_top3 = train_top3_model(X_train, y_train_encoded_hierarchial_rf)

# Train combination models
combination_models = train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes)

# Predict on test set
final_predictions = predict_with_top3_model(X_test, model_top3, combination_models, all_classes)
print("Final predictions:", final_predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded_hierarchial_rf, final_predictions)
print(f"Accuracy: {accuracy * 100:.2f}")  # Outputs accuracy as a percentage

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Step 1: Train the first Random Forest model for top-3 prediction
def train_top3_model(X_train, y_train_encoded_hierarchial_rf):
    model_top3 = RandomForestClassifier(random_state=42)
    model_top3.fit(X_train, y_train_encoded_hierarchial_rf)
    return model_top3

# Step 2: Create and train models for all combinations of top-3 classes
def train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes):
    combination_models = {}
    count = 0
    for combination in combinations(all_classes, 3):
        # Filter the dataset for the current combination
        mask = pd.Series(y_train_encoded_hierarchial_rf).isin(combination)
        X_subset = pd.DataFrame(X_train[mask], columns=X_train.columns)
        # print(X_subset)
        # break
        # print(X_subset)
        y_subset = y_train_encoded_hierarchial_rf[mask]
        count = count + 1
        print(count)
        if (X_subset.shape[0] != 0):
            # Step 6: Train an SVM model for the top 3 classes
            svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)  # You can change the kernel if needed
            svm_model.fit(X_subset, y_subset)
            combination_models[combination] = svm_model
    return combination_models

# Step 3: Top-3 prediction and refinement
def predict_with_top3_model(X_test, model_top3, combination_models, all_classes):
    # Step 3.1: Predict top-3 classes using model_top4
    probs = model_top3.predict_proba(X_test)
    top3_predictions = np.argsort(probs, axis=1)[:, -3:]  # Indices of top-3 classes

    final_predictions = []
    for i in range(len(X_test)):
        # Step 3.2: Get the actual top-3 class labels
        top3_classes = [all_classes[j] for j in top3_predictions[i]]
        top3_classes_tuple = tuple(sorted(top3_classes))  # Match the order in combination_models

        # Step 3.3: Use the corresponding combination model
        if top3_classes_tuple in combination_models:
            svm_model = combination_models[top3_classes_tuple]
            final_predictions.append(svm_model.predict(pd.DataFrame([X_test.iloc[i]], columns=X_test.columns))[0])
        else:
            # If no specific model is available, default to the top-3 class with highest probability
            final_predictions.append(top3_classes[-1])  # Most likely class
    return final_predictions

all_classes = sorted(list(set(y_train_encoded_hierarchial_rf)))  # Unique classes (14 in this case)

# Train the top-3 model
model_top3 = train_top3_model(X_train, y_train_encoded_hierarchial_rf)

# Train combination models
combination_models = train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes)

# Predict on test set
final_predictions = predict_with_top3_model(X_test, model_top3, combination_models, all_classes)
print("Final predictions:", final_predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded_hierarchial_rf, final_predictions)
print(f"Accuracy: {accuracy * 100:.2f}")  # Outputs accuracy as a percentage

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Step 1: Train the first Random Forest model for top-3 prediction
def train_top3_model(X_train, y_train_encoded_hierarchial_rf):
    model_top3 = RandomForestClassifier(random_state=42)
    model_top3.fit(X_train, y_train_encoded_hierarchial_rf)
    return model_top3

# Step 2: Create and train models for all combinations of top-3 classes
def train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes):
    combination_models = {}
    count = 0
    for combination in combinations(all_classes, 3):
        # Filter the dataset for the current combination
        mask = pd.Series(y_train_encoded_hierarchial_rf).isin(combination)
        X_subset = pd.DataFrame(X_train[mask], columns=X_train.columns)
        # print(X_subset)
        # break
        # print(X_subset)
        y_subset = y_train_encoded_hierarchial_rf[mask]
        count = count + 1
        print(count)
        if (X_subset.shape[0] != 0):
            # Define the Neural Network (MLPClassifier)
            mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42, activation='relu', solver='adam')
            mlp_model.fit(X_subset, y_subset)
            combination_models[combination] = mlp_model
    return combination_models

# Step 3: Top-3 prediction and refinement
def predict_with_top3_model(X_test, model_top3, combination_models, all_classes):
    # Step 3.1: Predict top-3 classes using model_top4
    probs = model_top3.predict_proba(X_test)
    top3_predictions = np.argsort(probs, axis=1)[:, -3:]  # Indices of top-3 classes

    final_predictions = []
    for i in range(len(X_test)):
        # Step 3.2: Get the actual top-3 class labels
        top3_classes = [all_classes[j] for j in top3_predictions[i]]
        top3_classes_tuple = tuple(sorted(top3_classes))  # Match the order in combination_models

        # Step 3.3: Use the corresponding combination model
        if top3_classes_tuple in combination_models:
            mlp_model = combination_models[top3_classes_tuple]
            final_predictions.append(mlp_model.predict(pd.DataFrame([X_test.iloc[i]], columns=X_test.columns))[0])
        else:
            # If no specific model is available, default to the top-3 class with highest probability
            final_predictions.append(top3_classes[-1])  # Most likely class
    return final_predictions

all_classes = sorted(list(set(y_train_encoded_hierarchial_rf)))  # Unique classes (14 in this case)

# Train the top-3 model
model_top3 = train_top3_model(X_train, y_train_encoded_hierarchial_rf)

# Train combination models
combination_models = train_combination_models(X_train, y_train_encoded_hierarchial_rf, all_classes)

# Predict on test set
final_predictions = predict_with_top3_model(X_test, model_top3, combination_models, all_classes)
print("Final predictions:", final_predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded_hierarchial_rf, final_predictions)
print(f"Accuracy: {accuracy * 100:.2f}")  # Outputs accuracy as a percentage

# Commented out IPython magic to ensure Python compatibility.
# %pip install shap

import shap
# Create a SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)
# Summary plot
shap.summary_plot(shap_values, X_test)
# Dependence plot for a specific feature (e.g., "RM")
# shap.dependence_plot("RM", shap_values, X_test)
# Force plot for a single prediction
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_test.iloc[0])
# Mean absolute SHAP values for global importance
importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print(feature_importance)

feature_importances = xgb_model.feature_importances_
print(feature_importances)
important_features = np.argsort(feature_importances)[::-1][:50]  # Top 50 features
print(important_features)
print(X_train.columns[important_features])
X_train_most_important_features = X_train.iloc[:, important_features]
X_test_most_important_features = X_test.iloc[:, important_features]

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Define the parameter grid for XGBoost
param_distributions = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [0.1, 1, 10],
}

# Initialize the XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random combinations to try
    scoring="f1_macro",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42,
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train_encoded)

# Display the best hyperparameters
print("Best Parameters:", random_search.best_params_)

# Use the best model to predict on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Commented out IPython magic to ensure Python compatibility.
# %pip install tensorflow

label_encoder2 = LabelEncoder()
label_encoder2.fit(y_train)
y_train_neural_network = label_encoder2.transform(y_train)
y_test_neural_network = label_encoder2.transform(y_test)

"""##**Normal Neural Network**"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(32, input_dim=106, activation='relu'))
model.add(Dense(16, activation='relu'))  # Hidden layer
num_classes = len(label_encoder2.classes_)
model.add(Dense(num_classes, activation='softmax'))  # Output layer for binary classification

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_neural_network, epochs=50, batch_size=10, validation_data=(X_test, y_test_neural_network))

loss, accuracy = model.evaluate(X_test, y_test_neural_network)
print(f'Accuracy: {accuracy}')

# Commented out IPython magic to ensure Python compatibility.
# %pip install optuna

"""##**Hyperparameter Tuning of Neural Network using Optuna**"""

import optuna

def create_model(trial):
    model1 = Sequential()
    model1.add(Dense(trial.suggest_int('units1', 16, 128), activation='relu', input_shape=(X_train.shape[1],)))
    model1.add(Dense(trial.suggest_int('units2', 16, 128), activation='relu'))
    num_classes = len(label_encoder2.classes_)
    model1.add(Dense(num_classes, activation='softmax'))

    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    model1.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model1

def objective(trial):
    model1 = create_model(trial)
    history = model1.fit(X_train, y_train_neural_network, validation_data=(X_test, y_test_neural_network), epochs=trial.suggest_int('epochs', 10, 50), batch_size=trial.suggest_int('batch_size', 10, 100), verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Visualization (optional)
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

from tensorflow.keras.optimizers import RMSprop
model1_optimal = Sequential()
model1_optimal.add(Dense(60, input_dim=106, activation='relu'))
model1_optimal.add(Dense(112, activation='relu'))  # Hidden layer
num_classes = len(label_encoder2.classes_)
model1_optimal.add(Dense(num_classes, activation='softmax'))  # Output layer for binary classification

model1_optimal.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model1_optimal.fit(X_train, y_train_neural_network, epochs=35, batch_size=38, validation_data=(X_test, y_test_neural_network))

loss, accuracy = model1_optimal.evaluate(X_test, y_test_neural_network)
print(f'Accuracy: {accuracy}')

"""## **Convolutional Neural Network**"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Step 1: Manually load or upload the CSV files
# Uncomment the next line if you're using Google Colab for file uploading
# uploaded = files.upload()

# # Step 2: Read the data from the separate CSV files (assuming you have uploaded them)
# X_train = pd.read_csv('X_train.csv').values  # Load X_train from the first CSV file
# y_train = pd.read_csv('y_train.csv').values  # Load Y_train from the second CSV file
# X_test = pd.read_csv('X_test.csv').values
# y_test = pd.read_csv('y_test.csv').values

# Step 3: Check the shapes of X_train and Y_train
print("X_train shape:", X_train_cnn.shape)
print("y_train shape:", y_train_cnn.shape)

# # Ensure X_train and Y_train have the same number of rows (samples)
# if (X_train.shape[0] != y_train.shape[0]):
#     print(f"Warning: Mismatch in number of samples. X_train has {X_train.shape[0]} samples and Y_train has {Y_train.shape[0]} samples.")
#     # Optionally, you can crop the larger dataset to match the smaller one
    # min_samples = min(X_train.shape[0], y_train.shape[0])
    # X_train = X_train[:min_samples]
    # y_train = y_train[:min_samples]

# Step 4: Normalize/standardize the feature data
scaler = MinMaxScaler()
X_train_normalized_cnn = scaler.fit_transform(X_train_cnn)

# Step 5: Convert Y_train (if it's categorical) into numeric labels
y_train_cnn = y_train_cnn.astype(str)
encoder = LabelEncoder()
# y_train_cnn = y_train_cnn.flatten()
# y_train_cnn = [y_train_cnn[i] for i in range(0, 2 * X_train_cnn.shape[0]) if (i % 2) == 1]
print(y_train_cnn)
print(len(y_train_cnn))
y_train_encoded_cnn = encoder.fit_transform(y_train_cnn)
# y_train_encoded = y_train_encoded[1]

# Step 6: Reshape X_train into 10x11 images
n_samples, n_features = X_train_normalized_cnn.shape
n_rows = 10
n_cols = 11  # We now have 110 features after padding

# Pad if necessary (ensure 110 features)
if n_features < n_rows * n_cols:
    padding = n_rows * n_cols - n_features
    X_train_normalized_cnn = np.pad(X_train_normalized_cnn, ((0, 0), (0, padding)), mode='constant', constant_values=0)

X_train_images = X_train_normalized_cnn.reshape(n_samples, n_rows, n_cols, 1)

# Verify the shape of reshaped data
print("X_train reshaped shape:", X_train_images.shape)

# Step 7: Create a directory to save the images
output_dir = 'output_images/train'
os.makedirs(output_dir, exist_ok=True)

# Step 8: Save each image as a PNG file
for i in range(n_samples):
    img = X_train_images[i].reshape(n_rows, n_cols)  # Reshape to 10x11 for visualization
    plt.imshow(img, cmap='gray', interpolation='nearest')  # Display image in grayscale
    plt.axis('off')  # Turn off axis
    plt.savefig(f"{output_dir}/image_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to avoid display
    if (i == 5):
        break

print(f"Images saved in {output_dir} directory.")

X_test_normalized_cnn = scaler.transform(X_test_cnn)
y_test_cnn = y_test_cnn.astype(str)
# y_test = y_test.flatten()
# y_test = [y_test[i] for i in range(0, 2 * X_test.shape[0]) if (i % 2) == 1]
print(y_test_cnn)
print(len(y_test_cnn))
y_test_encoded_cnn = encoder.transform(y_test_cnn)
# y_test_encoded = y_test_encoded[1]
# Pad if necessary (ensure 110 features)
if n_features < n_rows * n_cols:
    padding = n_rows * n_cols - n_features
    X_test_normalized_cnn = np.pad(X_test_normalized_cnn, ((0, 0), (0, padding)), mode='constant', constant_values=0)
# Step 6: Reshape X_train into 10x11 images
n_samples, n_features = X_test_normalized_cnn.shape
n_rows = 10
n_cols = 11  # We now have 110 features after padding
X_test_images = X_test_normalized_cnn.reshape(n_samples, n_rows, n_cols, 1)

# Verify the shape of reshaped data
print("X_test reshaped shape:", X_test_images.shape)

# Step 7: Create a directory to save the images
output_dir = 'output_images/test'
os.makedirs(output_dir, exist_ok=True)

# Step 8: Save each image as a PNG file
for i in range(n_samples):
    img = X_train_images[i].reshape(n_rows, n_cols)  # Reshape to 10x11 for visualization
    plt.imshow(img, cmap='gray', interpolation='nearest')  # Display image in grayscale
    plt.axis('off')  # Turn off axis
    plt.savefig(f"{output_dir}/image_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to avoid display
    if (i == 5):
        break

print(f"Images saved in {output_dir} directory.")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
y_train_encoded_cnn_array = np.array(y_train_encoded_cnn)
y_test_encoded_cnn_array = np.array(y_test_encoded_cnn)
y_train_encoded_cnn_array_one_hot = to_categorical(y_train_encoded_cnn_array, num_classes=14)
y_test_encoded_cnn_array_one_hot = to_categorical(y_test_encoded_cnn_array, num_classes=14)
# Define the CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output and add Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularization

        # Output Layer
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    return model

# Parameters
input_shape = (10, 11, 1)  # Assuming images are 11x10 with 1 channel
num_classes = 14           # Number of output classes
batch_size = 32            # Training batch size
epochs = 32                # Number of epochs

# Build the model
cnn_model = build_cnn(input_shape, num_classes)
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(
    X_train_images, y_train_encoded_cnn_array_one_hot,
    validation_data=(X_test_images, y_test_encoded_cnn_array_one_hot),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)
# # Step 5: Build the CNN model
# model = Sequential()

# # First convolutional layer with a smaller kernel size to avoid downsampling issues
# model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(10, 11, 1)))  # Adjust input shape if needed
# model.add(MaxPooling2D((2, 1)))  # Adjust pool size to avoid downsampling issues

# # Second convolutional layer with a smaller kernel size
# model.add(Conv2D(64, (3, 1), activation='relu'))
# model.add(MaxPooling2D((2, 1)))  # Adjust pool size to avoid downsampling issues

# # Flatten the output for the fully connected layers
# model.add(Flatten())

# # Fully connected layer
# model.add(Dense(128, activation='relu'))

# # Output layer (using softmax for multi-class classification)
# model.add(Dense(14, activation='softmax'))

# # Step 6: Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 7: Train the model
# history = model.fit(X_train_images, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_images, y_test_encoded))

# Step 8: Evaluate the model on the validation set
loss, accuracy = cnn_model.evaluate(X_test_images, y_test_encoded_cnn_array_one_hot)
print(f"Accuracy: {accuracy}")

# Step 9: Plot the training and validation accuracy/loss
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

df1 = pd.DataFrame(pd.read_excel('Features_SIH_Testing_Dataset_Initial_55.xlsx'))
df2 = pd.DataFrame(pd.read_excel('Features_SIH_Testing_Dataset_Final_30.xlsx'))
df1

df2

df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)
print(common_columns)

df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)'], inplace=True)
df = pd.concat([df1, df2], axis=1)
df

df.drop(columns=['Encrypted Data (Hex)', 'iv'], inplace=True)
X_SIH_Testing_Dataset = df

X_SIH_Testing_Dataset

X_SIH_Testing_Dataset.isnull().sum()

for i in X_SIH_Testing_Dataset:
    if X_SIH_Testing_Dataset[i].isnull().sum() != 0:
        print("Null value exists")

# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_SIH_Testing_Dataset['block_size'] = imputer.fit_transform(X_SIH_Testing_Dataset[['block_size']])
# X_SIH_Testing_Dataset.isnull().sum()

label_encoder1.classes_

X_SIH_Testing_Dataset['mode'] = label_encoder1.transform(X_SIH_Testing_Dataset[['mode']])
X_SIH_Testing_Dataset['mode']

for i in X_SIH_Testing_Dataset:
    if X_SIH_Testing_Dataset[i].dtype == 'object':
        print(i)

columns = []
for i in X_SIH_Testing_Dataset:
    if X_SIH_Testing_Dataset[i].dtype == 'object':
        columns.append(i)
X_SIH_Testing_Dataset.drop(columns=columns, inplace=True)
X_SIH_Testing_Dataset

exclude_columns = ['mode']
columns_to_scale = X_SIH_Testing_Dataset.columns.difference(exclude_columns)
X_SIH_Testing_Dataset[columns_to_scale] = scaler.transform(X_SIH_Testing_Dataset[columns_to_scale])
X_SIH_Testing_Dataset

"""##**Prediction using Random Forest Model**"""

y_prediction_SIH_Testing_Dataset = rf_model.predict(X_SIH_Testing_Dataset)
print(y_prediction_SIH_Testing_Dataset)

"""##**Prediction using Normal Neural Network**"""

import numpy as np
y_prediction_SIH_Testing_Dataset_neural_network = model.predict(X_SIH_Testing_Dataset)
y_prediction_SIH_Testing_Dataset = label_encoder2.inverse_transform([np.argmax(x) for x in y_prediction_SIH_Testing_Dataset_neural_network])
print(y_prediction_SIH_Testing_Dataset)

"""##**Prediction using Optimised Neural Network (Optuna)**"""

y_prediction_SIH_Testing_Dataset_neural_network = model1_optimal.predict(X_SIH_Testing_Dataset)
y_prediction_SIH_Testing_Dataset = label_encoder2.inverse_transform([np.argmax(x) for x in y_prediction_SIH_Testing_Dataset_neural_network])
print(y_prediction_SIH_Testing_Dataset)

