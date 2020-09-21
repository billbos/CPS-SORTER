import random
import json
import os
import csv
from cps_sorter.services.road_transformer import RoadTransformer
from cps_sorter.services.weka_helper import WekaHelper
class ModelEvaluator():
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.road_transformer = RoadTransformer()
        self.weka_helper = WekaHelper()

    def create_dataset(self, data_location, dataset_name):
        dataset = []
        for test_file in os.listdir(data_location):
            try:
                with open('{}/{}'.format(data_location, test_file)) as json_file:
                    test = json.load(json_file)
                    features = self.road_transformer.extract_features(test)
                    if test['execution']['oobs'] > 0:
                        features['safety'] = 'unsafe'
                    else:
                        features['safety'] = 'safe'
                    dataset.append(features)
            except Exception as e:
                print('test file {}: {}'.format(test_file, e))
                continue
        self._write_data_file('{}/{}_Complete.csv'.format(self.output_folder, dataset_name), dataset)
        return dataset

    def create_trainig_and_test_set(self, ratio, dataset_name):
        num_sample = int(len(self.complete_dataset)*ratio)
        training_set, test_set = self.rebalancing(self.complete_dataset, ratio)
        self.trainings_file = self._write_data_file('{}/{}_{}_training.csv'.format(self.output_folder, '{}-{}-split'.format(ratio, 1-ratio), dataset_name), training_set)
        self.test_file = self._write_data_file('{}/{}_{}_test.csv'.format(self.output_folder, '{}-{}-split'.format(ratio, 1-ratio), dataset_name), test_set)


    def rebalancing(self, training_set, ratio):
        safe = []
        unsafe = []
        for item in self.complete_dataset:
            if item['safety'] == 'safe':
                safe.append(item)
            else:
                unsafe.append(item)
        if len(safe) < len(unsafe):
            num_safe_sample = int(ratio *len(safe))
            num_unsafe_sample = int(ratio *len(unsafe))
            random.shuffle(safe)
            random.shuffle(unsafe)
            training_set = safe[:num_safe_sample] + unsafe[:num_safe_sample]
            test_set = safe[num_safe_sample:] + unsafe[num_unsafe_sample:]
        else:
            num_safe_sample = int(ratio *len(safe))
            num_unsafe_sample = int(ratio *len(unsafe))
            random.shuffle(safe)
            random.shuffle(unsafe)
            training_set = safe[:num_unsafe_sample] + unsafe[:num_unsafe_sample]
            test_set = safe[num_safe_sample:] + unsafe[num_unsafe_sample:]
        random.shuffle(training_set[1:])
        random.shuffle(test_set[1:])
        return training_set, test_set

    def _write_data_file(self, filename, rows=[]):
        with open(filename, 'w', newline='') as csv_file:
            fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
            'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'safety']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return filename


    def evaluate_models(self, dataset_name, data_location, ratios=[0.4,0.5,0.6,0.8]):
        result_file = '{}/{}_result.csv'.format(self.output_folder, dataset_name)
        with open(result_file, 'w', newline='') as csv_file:
            fieldnames = ['Model', 'Split', 'Training Accuracy', 'Test Accuracy', 'Cross-Validation', 
            'Precision Safe', 'Precision Unsafe', 'Recall Safe', 'Recall Unsafe', 'F-Measure Safe', 'F-Measure Unsafe',
            'Cros Precision Safe', 'Cros Precision Unsafe', 'Cros Recall Safe', 'Cros Recall Unsafe', 'Cros F-Measure Safe', 'Cros F-Measure Unsafe',
            'TPos Safe', 'TNeg Safe', 'FPos Safe', 'FNeg Safe', 'TPos Unsafe', 'TNeg Unsafe', 'FPos Unsafe', 'FNeg Unsafe',
            'Cros TPos Safe', 'Cros TNeg Safe', 'Cros FPos Safe', 'Cros FNeg Safe', 'Cros TPos Unsafe', 'Cros TNeg Unsafe', 'Cros FPos Unsafe', 'Cros FNeg Unsafe', 'Total Number'
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
        self.complete_dataset = self.create_dataset(data_location, dataset_name)
        for ratio in ratios:
            self.create_trainig_and_test_set(ratio, dataset_name)
            self.weka_helper.evaluate_models(str(ratio), self.trainings_file, self.test_file, result_file)
        return '{}_result'.format(dataset_name)
