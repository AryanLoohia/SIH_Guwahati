import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Example data (assuming you have a DataFrame with the extracted features)
df1 = pd.DataFrame(pd.read_excel('Features_Training_Initial_55.xlsx'))
df21 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_1.xlsx'))
df22 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_2.xlsx'))
df23 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_3.xlsx'))
df24 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_4.xlsx'))
df25 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_5.xlsx'))
df26 = pd.DataFrame(pd.read_excel('Splitted_Features_Training_Final_30_6.xlsx'))
df2 = pd.concat([df21, df22, df23, df24, df25, df26], axis=0)


df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)


df2 = df2.reset_index(drop=True)

df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)', 'Encrypted Data (Binary)', 'Original Text', 'Length', 'Algorithm'], inplace=True)
df = pd.concat([df1, df2], axis=1)


import numpy as np

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



# count = 0
# total = 0
# for i in range(0, len(df['Algorithm'])):
#     if (df['Algorithm'][i] == 'AES'):
#         total = total + 1
#         # print(df['block_size'][i])
#         if (df['block_size'][i] == 16):
#             count = count + 1
# print(count / total)



y_train = df['Algorithm']
df.drop(columns=['Original Text', 'Length', 'Encrypted Data (Binary)', 'Encrypted Data (Hex)', 'Algorithm', 'iv'], inplace=True)
X_train = df



"""#**Input (Testing)**

"""

# Example data (assuming you have a DataFrame with the extracted features)
df1 = pd.DataFrame(pd.read_excel('Features_Testing_Initial_55.xlsx'))
df2 = pd.DataFrame(pd.read_excel('Features_Testing_Final_30.xlsx'))


df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)


df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)', 'Encrypted Data (Binary)', 'Original Text', 'Length', 'Algorithm'], inplace=True)
df = pd.concat([df1, df2], axis=1)




import numpy as np

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

"""##Preparing Testing Dataset"""

y_test = df['Algorithm']
df.drop(columns=['Original Text', 'Length', 'Encrypted Data (Binary)', 'Encrypted Data (Hex)', 'Algorithm', 'iv'], inplace=True)
X_test = df


"""#**Data Preprocessing and Feature Engineering (from Object Columns) for Training Dataset**"""





# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_train['block_size'] = imputer.fit_transform(X_train[['block_size']])
# X_train.isnull().sum()

from sklearn.preprocessing import LabelEncoder

label_encoder1 = LabelEncoder()
label_encoder1.fit(X_train[['mode']])
def label_encoder1_function():
    return label_encoder1
X_train['mode'] = label_encoder1.transform(X_train[['mode']])




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




X_train.drop(columns=['byte_value_histogram'], inplace=True)



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




X_train.drop(columns='byte_value_percentiles', inplace=True)





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




X_train.drop(columns=['freq_byte_value_diff'], inplace=True)



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




X_train.drop(columns=['run_length_encoding'], inplace=True)



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


X_train.drop(columns=['byte_value_transition_matrix'], inplace=True)



# X_train.drop(columns=['byte_value_transition_matrix'], inplace=True)

X_train.drop(columns=['freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams'], inplace=True)



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





X_train.drop(columns=['byte_value_acf'], inplace=True)


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



X_train.drop(columns=['byte_value_power_spectrum'], inplace=True)



"""##Feature List

"""


features=list(X_train.columns)


"""##Imputing Null Values in Training Dataset"""



import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_train['byte_value_transition_matrix_entropy'] = imputer.fit_transform(X_train[['byte_value_transition_matrix_entropy']])

def byte_value_transition_matrix_entropy_imputer():
    return imputer



"""##Extracting Training Dataset for CNN"""

X_train_cnn = X_train
y_train_cnn = y_train

"""##Normalization of Training Dataset"""

from sklearn.preprocessing import StandardScaler
exclude_columns = ['mode', 'block_size', 'block_cipher_boolean', 'block_frequency', 'length', 'byte_distribution_uniformity_score', 'byte_distribution_low_frequency_byte_count', 'byte_distribution_skewness', 'byte_distribution_kurtosis', 'byte_distribution_dominant_byte_frequency', 'byte_distribution_byte_range_spread']
columns_to_scale = X_train.columns.difference(exclude_columns)

scaler = StandardScaler()


scaler.fit(X_train[columns_to_scale])
X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])

def scaler_function():
    return scaler

"""#**Data Preprocessing and Feature Engineering (from Object Columns) for Testing Dataset**"""



# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# X_test['block_size'] = imputer.fit_transform(X_test[['block_size']])
# X_test.isnull().sum()

X_test['mode'] = label_encoder1.transform(X_test[['mode']])


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


X_test.drop(columns=['byte_value_histogram'], inplace=True)

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


X_test.drop(columns='byte_value_percentiles', inplace=True)



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


X_test.drop(columns=['freq_byte_value_diff'], inplace=True)



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


X_test.drop(columns=['run_length_encoding'], inplace=True)


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


X_test.drop(columns=['byte_value_transition_matrix'], inplace=True)



X_test.drop(columns=['freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams'], inplace=True)



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


X_test.drop(columns=['byte_value_acf'], inplace=True)



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



X_test.drop(columns=['byte_value_power_spectrum'], inplace=True)



# columns = []
# for i in X_test:
#     if X_test[i].dtype == 'object':
#         columns.append(i)
# X_test.drop(columns=columns, inplace=True)
# X_test

"""##Imputing Null Values for Testing Dataset"""



X_test['byte_value_transition_matrix_entropy'] = imputer.transform(X_test[['byte_value_transition_matrix_entropy']])




import numpy as np
from sklearn.impute import SimpleImputer

imputer_sc = SimpleImputer(missing_values=np.nan, strategy='median')
X_test['serial_correlation'] = imputer_sc.fit_transform(X_test[['serial_correlation']])


"""##Exporting Final X_test and y_test Without Normalization"""



"""##Extracting Testing Dataset for CNN"""

X_test_cnn = X_test
y_test_cnn = y_test

"""##Normalization of Testing Dataset"""

exclude_columns = ['mode', 'block_size', 'block_cipher_boolean', 'block_frequency', 'length', 'byte_distribution_uniformity_score', 'byte_distribution_low_frequency_byte_count', 'byte_distribution_skewness', 'byte_distribution_kurtosis', 'byte_distribution_dominant_byte_frequency', 'byte_distribution_byte_range_spread']
columns_to_scale = X_test.columns.difference(exclude_columns)
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])





"""##**Random Forest Model**"""

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

pickle.dump(rf_model, open("model1.pkl", "wb"))