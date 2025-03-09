import numpy as np
import torch
import numpy as np
import math

np.random.seed(133)


def encode_map(input_array):
    p_map = {}
    length = len(input_array)

    for index, ele in zip(range(1, length + 1), input_array):
        p_map[str(ele)] = index
    return p_map


def decode_map(encode_map):
    de_map = {}
    for k, v in encode_map.items():
        de_map[v] = k
    return de_map


def get_prefix_sequence(sequence):
    i = 0
    list_seq = []
    while i < len(sequence):
        list_temp = []
        j = 0
        while j < (len(sequence.iat[i, 0]) - 1):
            list_temp.append(sequence.iat[i, 0][0 + j])
            list_seq.append(list(list_temp))
            j = j + 1
        i = i + 1
    return list_seq


def get_prefix_sequence_label(sequence):
    i = 0
    list_seq = []
    list_label = []
    list_case = []
    while i < len(sequence):
        list_temp = []
        j = 0
        while j < (len(sequence.iat[i, 0]) - 1):
            list_temp.append(sequence.iat[i, 0][0 + j])
            list_seq.append(list(list_temp))
            list_label.append(sequence.iat[i, 0][j + 1])
            list_case.append(i)
            j = j + 1
        i = i + 1
    return list_seq, list_label, list_case


def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def create_activity_activity(sequence):
    i = 0
    activity_list_src = []
    activity_list_dst = []
    case_list = []

    while i < len(sequence):
        if len(sequence.iat[i, 0]) == 1:
            src = sequence.iat[i, 0]
            dst = sequence.iat[i, 0]
        else:
            src = sequence.iat[i, 0][:-1]
            dst = sequence.iat[i, 0][1:]

        activity_list_src.append(src)
        activity_list_dst.append(dst)
        case_list.append(i)
        i = i + 1

    return activity_list_src, activity_list_dst, case_list


def get_index_of_duplicate_elements(arr):
    indices_dict = {}

    for index, value in enumerate(arr):

        if value == 0:
            continue

        if value in indices_dict:
            indices_dict[value].append(index)

        else:
            indices_dict[value] = [index]

    indices_list = list(indices_dict.values())

    return indices_list


def count_cases_with_repeated_activities(sequences):
    repeated_case_count = 0

    for act_seq in sequences:
        seen_activities = set()
        has_repeated = False

        for activity in act_seq:
            if activity in seen_activities:
                has_repeated = True
                break
            seen_activities.add(activity)

        if has_repeated:
            repeated_case_count += 1

    return repeated_case_count


def calculate_entropy(activity_sequence, ignore_padding=0):
    filtered_activity = [activity for activity in activity_sequence if activity != ignore_padding]
    if len(filtered_activity) == 0:
        return 0

    unique, counts = np.unique(filtered_activity, return_counts=True)
    probabilities = counts / len(filtered_activity)

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_longest_consecutive_subsequence(activity_sequence, ignore_padding=0):
    filtered_activity = [activity for activity in activity_sequence if activity != ignore_padding]
    if len(filtered_activity) == 0:
        return 0

    max_length = 0
    current_length = 1
    previous_activity = filtered_activity[0]
    for activity in filtered_activity[1:]:
        if activity == previous_activity:
            current_length += 1
        else:
            current_length = 1
            previous_activity = activity
        if current_length > max_length:
            max_length = current_length
    return max_length
