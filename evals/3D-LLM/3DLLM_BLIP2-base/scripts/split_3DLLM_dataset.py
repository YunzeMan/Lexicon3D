import json
import random

def split_data(input_data_joint, train_file, test_file, test_ratio):
    # Read the JSON data
    train_data = []
    test_data = []

    for data in input_data_joint:
        # Determine every nth element to pick for the test set
        step = int(1 / test_ratio)
        test_data.extend([item for index, item in enumerate(data) if index % step == 0])
        train_data.extend([item for index, item in enumerate(data) if index % step != 0])

    # Print the length of the train and test data
    print("The length of the train data is: ", len(train_data))
    print("The length of the test data is: ", len(test_data))

    # Write training data
    with open(train_file, 'w') as file:
        json.dump(train_data, file, indent=4)

    # Write testing data
    with open(test_file, 'w') as file:
        json.dump(test_data, file, indent=4)


def preprocess_annotations_and_read(input_file, mapping_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    data_filtered = []
    print("==> The length of the data before filtering is: ", len(data))
    if 'data_part2_scene' in input_file:
        with open(mapping_file, 'r') as file:
            mapping = json.load(file)
        for item in data:
            scene_id = item['scene_id']
            if str(scene_id) in mapping:
                item['scene_id'] = mapping[str(scene_id)]
                data_filtered.append(item)
    else:
        data_filtered = data
    print("==> The length of the data after filtering is: ", len(data_filtered))
    
    return data_filtered


# Usage
input_file_v2 = 'dataset/3DLLM/pretraining/data_part2_scene_v2.json'
input_file_v3 = 'dataset/3DLLM/pretraining/data_part2_scene_v3.json'
input_file_v3_01 = 'dataset/3DLLM/pretraining/chat_val_v3.json'
input_file_v3_02 = 'dataset/3DLLM/pretraining/task_val_v3.json'

mapping_file_v2 = 'dataset/3DLLM/pretraining/final_scene_map_dict_scan_v2.json'
mapping_file_v3 = 'dataset/3DLLM/pretraining/final_scene_map_dict_scan_v3.json'


# input_file_data = preprocess_annotations_and_read(input_file_v2, mapping_file_v2)
input_file_data = preprocess_annotations_and_read(input_file_v3, mapping_file_v3)
input_file_v3_01_data = preprocess_annotations_and_read(input_file_v3_01, mapping_file_v3)
input_file_v3_02_data = preprocess_annotations_and_read(input_file_v3_02, mapping_file_v3)


input_data_joint = [input_file_data, input_file_v3_01_data, input_file_v3_02_data]


# train_file = 'dataset/3DLLM/pretraining/data_part2_scene_v2_3_train.json'
# test_file = 'dataset/3DLLM/pretraining/data_part2_scene_v2_3_test.json'
train_file = 'dataset/3DLLM/pretraining/data_part2_scene_v3_train.json'
test_file = 'dataset/3DLLM/pretraining/data_part2_scene_v3_test.json'

split_data(input_data_joint, train_file, test_file, test_ratio=0.1)
