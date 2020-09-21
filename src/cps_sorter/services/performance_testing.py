import json
import datetime
import random
import tempfile
import os.path
import pandas as pd
from os import listdir
from shutil import copyfile
from cps_sorter.services.weka_helper import WekaHelper
from cps_sorter.services.road_transformer import RoadTransformer


class PerformanceTester:
    def __init__(self, weka_helper):
        self.weka_helper = weka_helper
        self.road_transformer = RoadTransformer()


    def offline_testing(self, dataset_path):
        self.dataset_path = dataset_path

    def get_avg_results(self, results):
        avg_results = {
            'results': []
        }
        for result in results:
            avg_results['results'].append(result)
            for key, value in result.items():
                avg_results.setdefault('avg_{}'.format(key), 0)
                avg_results['avg_{}'.format(key)] += value / len(results)
        return avg_results

    def get_random_baseline_fixed_test_num(self, test_set, num_tests, rounds):
        results = []
        for i in range(0, rounds):
            test_files = self.random_test_selection(test_set, num_tests)
            results.append(self.evaluate_tests(test_files))
            print('Random_fixed. round: {}'.format(i))

        
        return self.get_avg_results(results)
    
    


    def random_test_selection(self, tests=[], num_tests=10):
        random_selected_tests = []
        sampled_test = 0
        while sampled_test < num_tests:
            random_num = random.randint(0, (len(tests)-1))
            if tests[random_num] not in random_selected_tests:
                random_selected_tests.append(tests[random_num])
                sampled_test += 1
        return random_selected_tests


    def get_random_baseline_reach_unsafe_num(self, test_set, num_unsafe, rounds):
        results = []
        for i in range(0, rounds):
            print('Random_reached. round: {}'.format(i))
            count_unsafe = 0
            test_files = []
            max_index = len(test_set)-1
            while(count_unsafe < num_unsafe):
                test = test_set[random.randint(0, max_index)]
                test_files.append(test)
                with open(test) as json_file:
                    data = json.load(json_file)
                    if data['execution']['oobs'] > 0:
                        count_unsafe += 1
            results.append(self.evaluate_tests(test_files))
        return self.get_avg_results(results)
   
    def evaluate_tests(self, tests):
        result = {
            'total_cost': 0,
            'cost_safe': 0,
            'cost_unsafe': 0,
            'num_unsafe':0,
            'num_safe': 0
        }
        for test in tests:
            with open(test) as json_file:
                data = json.load(json_file)
                execution_time = (datetime.datetime.strptime(data['execution']['end_time'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.datetime.strptime(data['execution']['start_time'], '%Y-%m-%dT%H:%M:%S.%f')).total_seconds()
                if data['execution']['oobs'] > 0:
                    result['num_unsafe'] += 1
                    result['cost_unsafe'] += execution_time

                else:
                    result['cost_safe'] += execution_time
                    result['num_safe'] += 1
                result['total_cost'] += execution_time
        result['num_tests'] = result['num_safe'] + result['num_unsafe']
        return result

    def model_based_fixed_baseline(self, test_set, num_tests, rounds, models=[]):
        model_to_result = {}
        for model in models:
            results = []
            for i in range(0, rounds):
                print('Model_fixed. round: {}'.format(i))
                results.append(self.round_model_performance_test(test_set, num_tests, model))
            model_to_result[model] = self.get_avg_results(results)

        return model_to_result

    def round_model_performance_test(self, test_set, num_tests, model):
        result = {
                'num_missed_unsafe_tests': 0,
                'num_missed_safe_tests': 0,
                'saved_costs': 0,
                'total_costs': 0,
                'num_safe_file_tested': 0,
                'num_unsafe_file_tested': 0,
                'cost_from_safe_file': 0
            }
        tested_files = 0
        already_tested_files = []
        while tested_files < num_tests:
            while True:
                file = self.random_test_selection(test_set, 1)[0]
                if file not in already_tested_files:
                    already_tested_files.append(file)
                    break
            test_file = self.road_transformer.convert_to_test(file, is_file=True)
            safety_prediction = self.weka_helper.make_prediction(model, test_file)
            # os.remove(test_file)

            ev = self.evaluate_test(file)
            if safety_prediction == 'safe':
                if ev['is_safe']:
                    result['saved_costs'] += ev['cost']
                    result['num_missed_safe_tests'] += 1
                else:
                    result['num_missed_unsafe_tests'] += 1

            elif safety_prediction == 'unsafe':
                tested_files += 1
                if ev['is_safe']:
                    result['cost_from_safe_file'] += ev['cost']
                    result['num_safe_file_tested'] += 1
                else:
                    result['num_unsafe_file_tested'] += 1
                result['total_costs'] += ev['cost']
        return result

    def get_model_baseline_reach_unsafe_num(self, test_set, num_unsafe, rounds, models):
        model_to_result = {}
        for model in models:
            results = []
            for i in range(0, rounds):
                print('Model_reached. round: {}'.format(i))
                count_unsafe = 0
                test_files = []
                max_index = len(test_set)-1
                result = {
                    'num_missed_unsafe_tests': 0,
                    'num_missed_safe_tests': 0,
                    'saved_costs': 0,
                    'total_costs': 0,
                    'num_safe_file_tested': 0,
                    'num_unsafe_file_tested': 0,
                    'cost_from_safe_file': 0
                }
                while(count_unsafe < num_unsafe):
                    while True:
                        test = test_set[random.randint(0, max_index)]
                        if test not in test_files:
                            test_files.append(test)
                            break

                    test_file = self.road_transformer.convert_to_test(test, is_file=True)

                    safety_prediction = self.weka_helper.make_prediction(model, test_file)
                    # os.remove(test_file)

                    ev = self.evaluate_test(test)
                    if safety_prediction == 'safe':
                        if ev['is_safe']:
                            result['saved_costs'] += ev['cost']
                            result['num_missed_safe_tests'] += 1
                        else:
                            result['num_missed_unsafe_tests'] += 1

                    elif safety_prediction == 'unsafe':
                        if ev['is_safe']:
                            result['cost_from_safe_file'] += ev['cost']
                            result['num_safe_file_tested'] += 1
                        else:
                            result['num_unsafe_file_tested'] += 1
                            count_unsafe += 1
                        result['total_costs'] += ev['cost']

                results.append(result)
            
            model_to_result[model] = self.get_avg_results(results)

        return model_to_result
    
    def evaluate_test(self, test):
        result = {
            'cost': 0,
            'is_safe': True,
        }
        with open(test) as json_file:
            data = json.load(json_file)
            execution_time = (datetime.datetime.strptime(data['execution']['end_time'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.datetime.strptime(data['execution']['start_time'], '%Y-%m-%dT%H:%M:%S.%f')).total_seconds()
            if data['execution']['oobs'] > 0:
                result['is_safe'] = False
            result['cost'] += execution_time
        return result


def split_data(safe_dir, unsafe_dir, out_dir, train_test_ratio, unsafe_ratio):
    counter = 0
    safe_files = ['{}/{}'.format(safe_dir, f) for f in listdir(safe_dir) if os.path.isfile(os.path.join(safe_dir, f))]
    unsafe_files = ['{}/{}'.format(unsafe_dir, f) for f in listdir(unsafe_dir) if os.path.isfile(os.path.join(unsafe_dir, f))]

    if len(safe_files) < len(unsafe_files):
        num_to_sample = int(train_test_ratio * len(safe_files))
    else:
        num_to_sample = int(train_test_ratio * len(unsafe_files))

    training_dir_path = '{}/training'.format(out_dir)
    test_dir_path = '{}/test'.format(out_dir)
    os.mkdir(training_dir_path)
    os.mkdir(test_dir_path)
    safe_training = random.sample(safe_files, num_to_sample)
    
    for safe_sample in safe_training:
        safe_files.remove(safe_sample)

    unsafe_training  = random.sample(unsafe_files, num_to_sample)

    for unsafe_sample in unsafe_training:
        unsafe_files.remove(unsafe_sample)
    training_set = safe_training + unsafe_training
    
    total_sum = len(unsafe_files) + len(safe_files)
    len_unsafe = len(unsafe_files)
    len_safe = len(safe_files)
    num_unsafe_sample = int(unsafe_ratio * total_sum)
    num_safe_sample = int((1-unsafe_ratio)*total_sum)

    if num_safe_sample > len_safe:
        num_safe_sample  = len_safe
        num_unsafe_sample = int(num_safe_sample / (1-unsafe_ratio) * unsafe_ratio)
        if num_unsafe_sample > len_unsafe:
            num_unsafe_sample = len_unsafe
            num_safe_sample = int(len_safe / unsafe_ratio * (1-unsafe_ratio))

    if num_unsafe_sample > len(unsafe_files):
        num_unsafe_sample = len_unsafe
        num_safe_sample  = int(num_unsafe_sample / unsafe_ratio * (1-unsafe_ratio))
        if num_safe_sample > len_safe:
            num_safe_sample = len_safe
            num_unsafe_sample = int(len_safe /  (1-unsafe_ratio) * unsafe_ratio)

    safe_test = random.sample(safe_files,num_safe_sample)
    unsafe_test = random.sample(unsafe_files,num_unsafe_sample)
    test_set = safe_test + unsafe_test
    for training in training_set:
        filename = training.split('/')[-1]
        copyfile(training, '{}/{}'.format(training_dir_path, filename))
    for test in test_set:
        filename = test.split('/')[-1]
        copyfile(test, '{}/{}'.format(test_dir_path, filename))
    
    print(count_safe_unsafe(test_dir_path))
    return training_dir_path, test_dir_path

def count_safe_unsafe(folder):
    result = {'unsafe':0, 'safe':0}
    for subdir, dirs, files in os.walk(folder):
        for c, filename in enumerate(files):
            is_timeout = False
            splited_subdir = subdir.split('\\')
            dir_name = splited_subdir[-1]
            filepath = subdir + os.sep + filename
            with open(filepath) as json_file:
                try:
                    data = json.load(json_file)
                    execution = data.pop('execution', {})
                    if execution['reason'] == 'timeout':
                        continue
                    if execution['oobs'] > 0:
                        result['unsafe'] += 1
                    else:
                        result['safe'] += 1
                
                except Exception as e:
                    print(e)
    return result

# if __name__ == '__main__':
#     # dataset = 'D:/MasterThesis/DataSet/performance_test/beamng'
#     safe_dir = 'D:/MasterThesis/DataSet/performance_test/beamng/safe'
#     unsafe_dir = 'D:/MasterThesis/DataSet/performance_test/beamng/unsafe'
#     output = 'D:/MasterThesis/Results/Performance/Offline'
#     road_transformer = RoadTransformer()
#     with tempfile.TemporaryDirectory() as temp_dir:
#         training_dir, test_dir = split_data(safe_dir, unsafe_dir, temp_dir, 0.8, 0.45)
        
#         weka_helper = WekaHelper()
#         data_file = '{}/{}'.format(temp_dir, 'data_set.csv')
#         data_file = road_transformer.transform_to_training_data(training_dir, data_file, 'default')
#         training_file = road_transformer.create_training_test(data_file, temp_dir)
#         weka_helper.build_models(trainings_file=training_file, temp_dir=temp_dir, models=['J48.model', 'RandomForest.model', 'Logistic.model'])
        
#         tester = PerformanceTester(weka_helper)
        
#         result = {}
#         tests = ['{}/{}'.format(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
#         result['random_fix'] = tester.get_random_baseline_fixed_test_num(test_set=tests, num_tests=10, rounds=30)
#         result['random_reach'] = tester.get_random_baseline_reach_unsafe_num(test_set=tests, num_unsafe=10, rounds=30)
#         # print(result)
       
#         models = weka_helper.get_models()
#         result['model_fix'] = tester.model_based_fixed_baseline(tests, num_tests=10, rounds=30, models=models)
#         result['model_reach'] = tester.get_model_baseline_reach_unsafe_num(tests, num_unsafe=10, rounds=30, models=models)
#         with open('{}/30_rounds_10_tests_ratio0_45.json'.format(output), 'w') as outfile:
#             outfile.write(json.dumps(result, sort_keys=True, indent=4))
#     print('{}/30_rounds_10_tests_ratio0_45.json'.format(output))
   