
import numpy as np
import pandas as pd
import pickle  # Assuming you're using joblib to load your model
from collections import Counter
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import special
from scipy.stats import skew, kurtosis, chisquare, kstest, entropy as scipy_entropy
from numpy.fft import fft
from scipy.stats import entropy, skew, kurtosis
import ast
import itertools



def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

# Calculate entropy of byte data
def calculate_entropy(data):
    if (len(data) == 0):
        return 0
    entropy = 0
    data_len = len(data)
    counter = Counter(data)
    for count in counter.values():
        probability = count / data_len
        entropy -= probability * math.log2(probability)
    return entropy

# Top-N frequencies
def top_n_frequencies(data, n=5):
    freq_dist = Counter(data)
    most_common = freq_dist.most_common(n)
    return {f'top_{i+1}_freq': count for i, (char, count) in enumerate(most_common)}

# Byte-level statistics
def byte_statistics(data):
    if len(data) == 0:  # Check if the input data is empty
        return {
            'mean_byte_value': 0,
            'median_byte_value': 0,
            'variance_byte_value': 0,
            'std_dev_byte_value': 0,
            'skewness_byte_value': 0,
            'kurtosis_byte_value': 0,
        }
    
    byte_values = np.array(list(data))
    stats = {
        'mean_byte_value': np.mean(byte_values),
        'median_byte_value': np.median(byte_values),
        'variance_byte_value': np.var(byte_values),
        'std_dev_byte_value': np.std(byte_values),
        'skewness_byte_value': skew(byte_values),
        'kurtosis_byte_value': kurtosis(byte_values),
    }
    return stats

# Frequency statistics
def frequency_statistics(data):
    freq_dist = Counter(data)
    freqs = np.array(list(freq_dist.values()))
    stats = {
        'max_byte_freq': np.max(freqs),
        'min_byte_freq': np.min(freqs),
        'range_byte_freq': np.max(freqs) - np.min(freqs),
        'std_dev_byte_freq': np.std(freqs),
        'entropy_byte_freq': scipy_entropy(list(freqs))  # Convert dict_values to list
    }
    return stats

# N-gram (Bigram, Trigram, Quadgram) statistics
def ngram_statistics(data, n=2):
    if len(data) < n:  # Guard condition to check if there's enough data
        return {f'{n}gram_max_freq': 0, f'{n}gram_min_freq': 0, f'{n}gram_range_freq': 0, f'{n}gram_std_dev_freq': 0, f'{n}gram_entropy_freq': 0}
    
    ngrams = Counter([tuple(data[i:i+n]) for i in range(len(data)-n+1)])
    freqs = np.array(list(ngrams.values()))
    if freqs.size == 0:  # Additional check for empty frequencies
        return {f'{n}gram_max_freq': 0, f'{n}gram_min_freq': 0, f'{n}gram_range_freq': 0, f'{n}gram_std_dev_freq': 0, f'{n}gram_entropy_freq': 0}
    
    stats = {
        f'{n}gram_max_freq': np.max(freqs),
        f'{n}gram_min_freq': np.min(freqs),
        f'{n}gram_range_freq': np.max(freqs) - np.min(freqs),
        f'{n}gram_std_dev_freq': np.std(freqs),
        f'{n}gram_entropy_freq': scipy_entropy(list(freqs))
    }
    return stats

# Calculate autocorrelation at a given lag
def calculate_autocorrelation(data, lag):
    byte_values = np.array(list(data))  # Convert byte data into a list of integers
    n = len(byte_values)
    mean = np.mean(byte_values)
    autocorr = np.correlate(byte_values - mean, byte_values - mean, mode='full')[n - 1:] / np.var(byte_values) / n
    return autocorr[lag] if lag < len(autocorr) else 0

# FFT statistics
def fft_statistics(data):
    byte_values = np.array(list(data), dtype=np.float64)  # Ensure proper data type
    if byte_values.size == 0:  # Guard condition for empty byte data
        return {
            'fft_mean_magnitude': 0,
            'fft_std_dev_magnitude': 0,
            'fft_max_magnitude': 0,
            'fft_min_magnitude': 0,
            'fft_median_magnitude': 0,
        }
    
    fft_vals = np.abs(fft(byte_values))
    return {
        'fft_mean_magnitude': np.mean(fft_vals),
        'fft_std_dev_magnitude': np.std(fft_vals),
        'fft_max_magnitude': np.max(fft_vals),
        'fft_min_magnitude': np.min(fft_vals),
        'fft_median_magnitude': np.median(fft_vals),
    }

# Calculate compression ratio (using gzip for example)
def compression_ratio(data):
    import gzip
    compressed = gzip.compress(data)
    return len(compressed) / len(data)

# Hamming weight of bytes
def average_hamming_weight(data):
    hamming_weight = sum(bin(byte).count('1') for byte in data)
    return hamming_weight / len(data)

# Run tests for randomness (based on consecutive bytes)
def runs_test(data):
    runs = 1
    for i in range(1, len(data)):
        if data[i] != data[i-1]:
            runs += 1
    return runs

# Chi-square test statistic
def chi_square_test(data):
    freq_dist = Counter(data)
    observed = np.array(list(freq_dist.values()))
    expected = np.full(len(observed), np.mean(observed))
    chi2, _ = chisquare(observed, expected)
    return chi2

# Kolmogorov-Smirnov test statistic (against uniform distribution)
def ks_test(data):
    byte_values = np.array(list(data))
    d_stat, _ = kstest(byte_values, 'uniform', args=(np.min(byte_values), np.max(byte_values)))
    return d_stat

# Serial correlation
def serial_correlation(data):
    byte_values = np.array(list(data))
    return np.corrcoef(byte_values[:-1], byte_values[1:])[0, 1]

# Percentage of printable ASCII characters
def printable_ascii_percentage(data):
    printable = sum(32 <= byte <= 126 for byte in data)
    return printable / len(data)

# Extract features from the ciphertext
def extract_features(ciphertext_hex, features):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    features['length'] = len(ciphertext_bytes)

    # Byte-level statistics
    byte_stats = byte_statistics(ciphertext_bytes)
    features.update(byte_stats)

    # Entropy
    features['entropy'] = calculate_entropy(ciphertext_bytes)

    # Frequency distribution statistics
    freq_stats = frequency_statistics(ciphertext_bytes)
    features.update(freq_stats)

    # Bigram, Trigram, and Quadgram statistics
    for n in [2, 3, 4]:
        ngram_stats = ngram_statistics(ciphertext_bytes, n=n)
        features.update(ngram_stats)

    # Autocorrelation
    for lag in [1, 2, 5, 10]:
        features[f'autocorr_lag_{lag}'] = calculate_autocorrelation(ciphertext_bytes, lag)

    # FFT statistics
    fft_stats = fft_statistics(ciphertext_bytes)
    features.update(fft_stats)

    # Compression ratio
    features['compression_ratio'] = compression_ratio(ciphertext_bytes)

    # Hamming weight
    features['avg_hamming_weight'] = average_hamming_weight(ciphertext_bytes)

    # Runs test statistic
    features['runs_test'] = runs_test(ciphertext_bytes)

    # Chi-square test
    features['chi_square_test'] = chi_square_test(ciphertext_bytes)

    # Kolmogorov-Smirnov test
    features['ks_test_stat'] = ks_test(ciphertext_bytes)

    # Serial correlation
    features['serial_correlation'] = serial_correlation(ciphertext_bytes)

    # Percentage of printable ASCII characters
    features['printable_ascii_percentage'] = printable_ascii_percentage(ciphertext_bytes)

    # Additional byte-level statistics
    byte_values = np.array(list(ciphertext_bytes))
    features['avg_byte_value_change'] = np.mean(np.abs(np.diff(byte_values)))
    features['median_abs_dev_byte_values'] = np.median(np.abs(byte_values - np.median(byte_values)))
    features['iqr_byte_values'] = np.percentile(byte_values, 75) - np.percentile(byte_values, 25)
    features['coef_variation_byte_values'] = np.std(byte_values) / np.mean(byte_values) if np.mean(byte_values) != 0 else 0
    features['pct_bytes_above_mean'] = np.sum(byte_values > np.mean(byte_values)) / len(byte_values)

    # Entropy of byte value gaps
    byte_value_gaps = np.abs(np.diff(byte_values))
    features['entropy_byte_value_gaps'] = scipy_entropy(list(Counter(byte_value_gaps).values()))  

    
    
    return features

# Extract IV and infer mode of operation
def extract_iv_and_infer_mode(ciphertext_hex, features, block_size=16):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    iv = ciphertext_bytes[:block_size]
    features['iv'] = iv

    if len(ciphertext_bytes) % block_size != 0:
        features['mode'] = 'Unknown or Stream Cipher'
    else:
        blocks = [ciphertext_bytes[i:i + block_size] for i in range(0, len(ciphertext_bytes), block_size)]
        if len(blocks) != len(set(blocks)):
            features['mode'] = 'ECB'
        else:
            features['mode'] = 'CBC or other block mode'

    return features

def byte_value_range(data):
    return np.ptp(data)

def mode_of_byte_values(data):
    return Counter(data).most_common(1)[0][0]

def frequency_of_mode_byte_value(data):
    return Counter(data).most_common(1)[0][1] / len(data)



def byte_value_percentiles(data):
    return np.percentile(data, [25, 50, 75]).tolist()

def entropy_of_byte_value_differences(data):
    if (len(data) != 0):
        differences = np.diff(data)
        return calculate_entropy(differences)
    else:
        return 0

# def frequency_of_byte_value_differences(data):
#     differences = np.diff(data)
#     return dict(Counter(differences))

def longest_increasing_subsequence(data):
    n = len(data)
    if n == 0:
        return 0
    lengths = [1] * n
    for i in range(1, n):
        for j in range(i):
            if data[i] > data[j] and lengths[i] < lengths[j] + 1:
                lengths[i] = lengths[j] + 1
    return max(lengths)

def longest_decreasing_subsequence(data):
    return longest_increasing_subsequence([-x for x in data])

# def run_length_encoding(data):
#     return [(len(list(group)), name) for name, group in itertools.groupby(data)]

# def byte_value_transition_matrix(data):
#     matrix = np.zeros((256, 256), dtype=int)
#     for i in range(len(data) - 1):
#         matrix[data[i]][data[i+1]] += 1
#     return matrix.tolist()

def frequency_of_byte_value_n_grams(data, n):
    n_grams = zip(*[data[i:] for i in range(n)])
    return dict(Counter(n_grams))

def entropy_of_byte_value_n_grams(data, n):
    n_gram_freq = frequency_of_byte_value_n_grams(data, n)
    return scipy_entropy(list(n_gram_freq.values()))

def byte_value_autocorrelation_function(data, nlags=50):
    result = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    result = result[result.size//2:]
    return result[:nlags].tolist()

def byte_value_power_spectrum(data):
    return np.abs(np.fft.fft(data))**2

# Updated extract_features function
def extract_features1_new(ciphertext_hex, features):
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    byte_values = np.array(list(ciphertext_bytes))

    # Existing feature extraction (keep all the existing feature extractions)

    # New feature extractions
    features['byte_value_range'] = byte_value_range(byte_values)
    features['mode_byte_value'] = mode_of_byte_values(byte_values)
    features['freq_mode_byte_value'] = frequency_of_mode_byte_value(byte_values)
    
    # features['byte_value_percentiles'] = byte_value_percentiles(byte_values)
    features['entropy_byte_value_diff'] = entropy_of_byte_value_differences(byte_values)
    # features['freq_byte_value_diff'] = frequency_of_byte_value_differences(byte_values)
    features['longest_increasing_subseq'] = longest_increasing_subsequence(byte_values)
    features['longest_decreasing_subseq'] = longest_decreasing_subsequence(byte_values)
    # features['run_length_encoding'] = run_length_encoding(byte_values)
    # features['byte_value_transition_matrix'] = byte_value_transition_matrix(byte_values)

    for n in [2, 3, 4]:
        features[f'freq_byte_value_{n}grams'] = frequency_of_byte_value_n_grams(byte_values, n)
        features[f'entropy_byte_value_{n}grams'] = entropy_of_byte_value_n_grams(byte_values, n)

    
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

    
        repeated_blocks = sum(count > 1 for count in block_counter.values())
        total_blocks = len(blocks)
        return repeated_blocks / total_blocks * 100
    
    features['block_size'] = get_block_size(ciphertext_hex)

    if features['block_size'] != 0:
        features['block_cipher_boolean'] = 1
    else:
        features['block_cipher_boolean'] = 0

    if features['block_size'] == 8:
        features['block_frequency'] = calculate_block_frequency(ciphertext_hex, features['block_size'])
    else:
        features['block_frequency'] = 0


    

# Function to convert Hex to Binary
    def hex_to_binary(ciphertext_hex):
        return ''.join(format(byte, '08b') for byte in bytes.fromhex(ciphertext_hex))

# Frequency Test (Monobit Test)
    def frequency_test(binary_sequence):
   
        n = len(binary_sequence)
        s = np.sum([int(bit) for bit in binary_sequence])
        s_obs = abs(s - n/2) / np.sqrt(n/4)

    # Calculate p-value
        p_value = math.erfc(s_obs / np.sqrt(2))
        return p_value

# Runs Test
    def runs_test(binary_sequence):
    
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
    binary_data = hex_to_binary(ciphertext_hex)

    
    freq_p_value = frequency_test(binary_data)
    runs_p_value = runs_test(binary_data)
    longest_run_p_value = longest_run_test(binary_data)

    # Combine all the extracted features into a dictionary
    features['nist_frequency_test_p_value'] = freq_p_value
    features['nist_runs_test_p_value'] = runs_p_value
    features['nist_longest_run_test_p_value'] = longest_run_p_value

    byte_data = [int(c, 16) for c in ciphertext_hex]
   
    def byte_value_histogram(data, bins=256):
        hist, _ = np.histogram(data, bins=bins, range=(0, 255))
        
        return hist.tolist()
    
    i = byte_value_histogram(byte_data)
    
    
    l = i
    l1 = []
    for j in l:
        
        l1.append(int(j))
        
    
    features['byte_value_histogram_mean'] = sum(l1) / len(l1)
    features['byte_value_histogram_std_dev'] = np.std(l1)
    
    total_sum = sum(l1)
    byte_distribution = [i / total_sum for i in l1]
    features['byte_distribution_entropy'] = entropy(byte_distribution, base=2)
    
    ideal_uniform = 1 / 256
    features['byte_distribution_uniformity_score'] = 1 - np.sum(np.abs(np.array(byte_distribution) - ideal_uniform)) / 2
    
    mean_frequency = np.mean(byte_distribution)
    peak_frequency = max(byte_distribution)
    features['byte_distribution_peak_to_mean_ratio'] = peak_frequency / mean_frequency
    
    features['byte_distribution_low_frequency_byte_count'] = sum(1 for freq in byte_distribution if freq < 0.001)
    features['byte_distribution_skewness'] = skew(byte_distribution)
    features['byte_distribution_kurtosis'] = kurtosis(byte_distribution)
    features['byte_distribution_dominant_byte_frequency'] = max(byte_distribution)
    features['byte_distribution_byte_range_spread'] = max(byte_distribution) - min(byte_distribution)
    
    
    
    def byte_value_percentiles(data):
        return np.percentile(data, [25, 50, 75]).tolist()
    
    byte_value_percentiles=byte_value_percentiles(byte_data)
    
    
    l = byte_value_percentiles
    l1 = []
    for j in l:
        
        l1.append(float(j))
    
    
    features['byte_value_25th_percentile'] = l1[0]
    features['byte_value_50th_percentile'] = l1[0]
    features['byte_value_75th_percentile'] = l1[0]
    
    
    def frequency_of_byte_value_differences(data):
        differences = np.diff(data)
        return dict(Counter(differences))

    i= frequency_of_byte_value_differences(byte_data)
    
    keys = list(i.keys())
    values = list(i.values())
    # Assign each feature to the dictionary directly
    features['freq_byte_value_diff_mean_keys'] = np.mean(keys) if keys else 0
    features['freq_byte_value_diff_mean_values'] = np.mean(values) if values else 0
    features['freq_byte_value_diff_weighted_mean'] = np.average(keys, weights=values) if values else 0
    features['freq_byte_value_diff_max_keys'] = max(keys) if keys else 0
    features['freq_byte_value_diff_max_values'] = max(values) if values else 0
    features['freq_byte_value_diff_std_dev_keys'] = np.std(keys) if keys else 0

    def run_length_encoding(data):
        return [(len(list(group)), name) for name, group in itertools.groupby(data)]
    byte_data = [int(c, 16) for c in ciphertext_hex]
     
    run_length = run_length_encoding(byte_data)
    runs = [r[0] for r in run_length]  # Length of consecutive values
    values = [r[1] for r in run_length]  # Values of consecutive bytes
    

    features['run_length_encoding_total_encoding_length'] = sum(runs)  # Total encoding length
    features['run_length_encoding_max_run_length'] = max(runs) if runs else 0  # Max run length
    features['run_length_encoding_mean_run_length'] = np.mean(runs) if runs else 0  # Mean run length
    features['run_length_encoding_std_dev_run_length'] = np.std(runs) if len(runs) > 1 else 0  # Std Dev of run lengths
    features['run_length_encoding_mean_value'] = np.mean(values) if values else 0  # Mean of the values
    features['run_length_encoding_std_dev_value'] = np.std(values) if len(values) > 1 else 0  # Std Dev of the values
    
    def byte_value_transition_matrix(data):
        matrix = np.zeros((256, 256), dtype=int)
        for i in range(len(data) - 1):
            matrix[data[i]][data[i+1]] += 1
        return matrix

    
    
    byte_data = [int(c, 16) for c in ciphertext_hex]
    transition_matrix = byte_value_transition_matrix(byte_data)
    sparsity = 1 - np.count_nonzero(transition_matrix) / (256 * 256)
    prob_matrix = transition_matrix / np.sum(transition_matrix)
    matrix_entropy =scipy_entropy(prob_matrix.flatten(), base=2)
    top_k_transitions = np.sort(transition_matrix.flatten())[::-1][:5]
    top_k_sum = np.sum(top_k_transitions)
    normalized_matrix = transition_matrix / (np.sum(transition_matrix, axis=1, keepdims=True) + 1e-10)
    mean_prob_per_row = np.mean(normalized_matrix, axis=1).mean()
    quadrant_1_sum = np.sum(transition_matrix[:128, :128])
    quadrant_2_sum = np.sum(transition_matrix[:128, 128:])
    quadrant_3_sum = np.sum(transition_matrix[128:, :128])
    quadrant_4_sum = np.sum(transition_matrix[128:, 128:])
    
    features['byte_value_transition_matrix_sparsity'] = sparsity
    features['byte_value_transition_matrix_entropy'] = matrix_entropy
    features['byte_value_transition_matrix_top_k_sum'] = top_k_sum
    features['byte_value_transition_matrix_mean_prob_per_row'] = mean_prob_per_row
    features['byte_value_transition_matrix_quadrant_1_sum'] = quadrant_1_sum
    features['byte_value_transition_matrix_quadrant_2_sum'] = quadrant_2_sum
    features['byte_value_transition_matrix_quadrant_3_sum'] = quadrant_3_sum
    features['byte_value_transition_matrix_quadrant_4_sum'] = quadrant_4_sum

    acf = byte_value_autocorrelation_function(byte_values)
    
    
    acf = np.array(acf, dtype=np.float64)
    if len(acf) < 2:
        
            mean_acf= np.nan,
            variance_acf= np.nan,
            max_acf=np.nan,
            lag_of_max_acf= np.nan
            
    else:
        acf_no_lag0 = acf[1:]

        mean_acf = np.mean(acf_no_lag0)
        variance_acf = np.var(acf_no_lag0)
        max_acf = np.max(acf_no_lag0)
        lag_of_max_acf = np.argmax(acf_no_lag0) + 1 
    
    features['mean_acf'] = mean_acf
    features['variance_acf'] = variance_acf
    features['max_acf'] = max_acf
    features['lag_of_max_acf'] = lag_of_max_acf
    
    def total_power(power_spectrum):
        return np.sum(power_spectrum)

    def peak_power(power_spectrum):
        return np.max(power_spectrum)

    def power_concentration(power_spectrum, top_n=3):
        sorted_spectrum = np.sort(power_spectrum)[::-1]
        return np.sum(sorted_spectrum[:top_n]) / np.sum(power_spectrum)
    byte_values = [int(c, 16) for c in hex_data]
    power_spectrum = byte_value_power_spectrum(byte_values)
    features['total_power'] = total_power(power_spectrum)
    features['peak_power'] = peak_power(power_spectrum)
    features['power_concentration'] = power_concentration(power_spectrum)
    
    
    return features



    

hex_data=input("Enter hex data: ")

hex_data = ''.join(hex_data.split())

        

features = {}
features = extract_features(hex_data, features=features)

        
features = extract_iv_and_infer_mode(hex_data, features)

        

df1 = pd.DataFrame([features])

        
features = {}
features = extract_features1_new(hex_data, features=features)

        
df2 = pd.DataFrame([features])

        

features_df = pd.concat([df1, df2], axis=1)
features_df.drop(columns=['iv',  'freq_byte_value_2grams', 
                                  'freq_byte_value_3grams', 'freq_byte_value_4grams', 
                                  ], inplace=True)

pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', 1000)         # Adjust display width
pd.set_option('display.max_colwidth', None)


X = features_df

column_headers = features_df.columns.tolist()

# Print the list of column headers

print(column_headers)
print(X)


