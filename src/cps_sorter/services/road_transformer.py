import numpy as np
import pandas as pd
import cps_sorter.services.utility as utility
import tempfile
import csv
import json
import os
from asfault.tests import RoadTest
import random

class RoadTransformer:

    label_map = {
        'unsafe': 0,
        'safe': 1
    }
    reverse_label_map = {
        0: 'unsafe',
        1: 'safe'
    }

    def convert_to_test(self, test, is_file=False, exclude_features=[]):
        to_test = tempfile.NamedTemporaryFile(delete=False)
        fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
        'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'safety']
        for feature in exclude_features:
            fieldnames.pop(feature, None)

        with open(to_test.name, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()          
            features = self.extract_features_for_test_case(test, is_file, exclude_features)
            # writer.writerow(features)
            features['safety'] = 'safe'
            writer.writerow(features)
            features['safety'] = 'unsafe'
            writer.writerow(features)

        return to_test.name


    def convert_to_test_bulk(self, tests, exclude_features=[]):
        to_test = tempfile.NamedTemporaryFile(delete=False)
        fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
        'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'safety']
        for feature in exclude_features:
            fieldnames.pop(feature, None)

        with open(to_test.name, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            counter = 0
            for test in tests:       
                features = self.extract_features_for_test_case(RoadTest.to_dict(test), False, exclude_features)
                if len(tests) == 1:
                    features['safety'] = 'safe'
                    writer.writerow(features)
                    features['safety'] = 'unsafe'
                    writer.writerow(features)
                else:
                    if counter % 2 == 1:
                        features['safety'] = 'safe'
                        writer.writerow(features)
                    else:
                        features['safety'] = 'unsafe'
                        writer.writerow(features)
                    counter += 1

        return to_test.name
    # def extract_features_for_test_case(self, data, exclude_features=[]):
    #     features = self.extract_features(data)
    #     for f in exclude_features:
    #         features.pop(f, None)
            
    #     return features


    def extract_features_for_test_case(self, test, is_file=False, exclude_features=[]):
        try:
            if is_file:
                with open(test) as json_file:
                    test = json.load(json_file)
            features = self.extract_features(test)
            for f in exclude_features:
                features.pop(f, None)        
            return features
        except Exception as e:
                print(e)

    def extract_features(self, data):
        angles = []
        path = data['path']
        path_used = []
        road_distance = 0
        pivot_offs = []
        points = {}
        l_turns = 0
        r_turns = 0
        straight = 0
        road_distance = 0

        for seg_id in path:
            seg = data['network']['nodes'][str(seg_id)]
            if seg['roadtype'] == 'r_turn':
                r_turns += 1
            elif seg['roadtype'] == 'l_turn':
                l_turns += 1
            elif seg['roadtype'] == 'straight':
                straight += 1

            if seg['angle'] < 0:
                seg['angle'] += 360
            if seg['angle'] > 0:
                angles.append(seg['angle'])
            if seg['pivot_off'] > 0:
                pivot_offs.append(seg['pivot_off'])

            points[seg_id] = utility.Point(seg['x'], seg['y'])
            path_used.append(seg_id)

        if not angles:
            pivot_offs.append(0)
 
        if not pivot_offs:
            pivot_offs.append(0)   

        for i in range(0, len(path_used)-1):
            road_distance += utility.get_distance(points[path_used[i]], points[path_used[i+1]])
        direct_distance = utility.get_distance(points[path_used[0]], points[path_used[-1]])



        result = {
            'direct_distance': direct_distance,
            'road_distance': road_distance,
            'num_l_turns': l_turns,
            'num_r_turns': r_turns,
            'num_straights': straight,
            'median_angle': np.median(angles),
            'total_angle': np.sum(angles),
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'max_angle': np.max(angles),
            'min_angle': np.min(angles),
            'median_pivot_off': np.median(pivot_offs),
            'mean_pivot_off': np.mean(pivot_offs),
            'std_pivot_off': np.std(pivot_offs),
            'max_pivot_off': np.max(pivot_offs),
            'min_pivot_off': np.min(pivot_offs),
        }
    

        return result

    def transform_to_training_data(self, directory, outputfile, ai_type='beamng'): 
        '''
        creates a csv file out of json files from beamng data
        '''
        file_pairs = utility.search_files(directory)
        counter = 0
        # outputfile = '{}/{}'.format(self.output_folder, outputfile)
        # outputfile = '{}_{}'.format(ai_type, outputfile)
        with open(outputfile, 'w', newline='') as csv_file:
            # fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
            # 'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'ai', 'safety']
            fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
            'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'safety']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            # ais = ['beamng', 'driver_ai']
            # for ai_type in ais:
            for filename, file_paths in file_pairs[ai_type].items():
                with open(file_paths['exec_file']) as json_file:
                    try:
                        data = json.load(json_file)
                        if not 'execution' in data.keys():
                            continue
                        test_data = self.extract_test_data(data)
                        print('file: {}'.format(counter))
                        # test_data['ai'] = ai_type
                        counter += 1
                        writer.writerow(test_data)
                    except Exception as e:
                        print(e)
            
        return outputfile

    def transform_tests_to_training_data(self, tests, outputfile, with_header=False): 
        '''
        creates a csv file out of json files from beamng data
        '''
        if with_header:
            mode = 'w'
        else:
            mode = 'a+'
        unsafe_tests = []
        safe_tests = []
        for test in tests:
            test = RoadTest.to_dict(test)
            features = self.extract_features(test)
            if test['execution']['oobs'] > 0:
                features['safety'] = 'unsafe'
                unsafe_tests.append(features)
            else:
                features['safety'] = 'safe'
                safe_tests.append(features)

        combined_tests = safe_tests[1:]+unsafe_tests
        random.shuffle(combined_tests)
        trainings_data = safe_tests[:1] + combined_tests
        with open(outputfile, mode, newline='') as csv_file:
            fieldnames = ['direct_distance', 'road_distance', 'num_l_turns','num_r_turns','num_straights','median_angle','total_angle','mean_angle','std_angle',
            'max_angle','min_angle','median_pivot_off','mean_pivot_off','std_pivot_off','max_pivot_off','min_pivot_off', 'safety']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if with_header:
                writer.writeheader()
            for feature in trainings_data:
                writer.writerow(feature)
                  
        return outputfile

    def extract_test_data(self, data):
        angles = []
        path = data['path']
        path_used = []
        road_distance = 0
        pivot_offs = []
        points = {}
        l_turns = 0
        r_turns = 0
        straight = 0
        road_distance = 0
        num_oobs = data['execution']['oobs']
        # road_test = RoadTest.from_dict(data)
        # dis = RoadTest.get_suite_coverage([road_test], 2)
        for seg_id in path:
            seg = data['network']['nodes'][str(seg_id)]
            if seg['roadtype'] == 'r_turn':
                r_turns += 1
            elif seg['roadtype'] == 'l_turn':
                l_turns += 1
            elif seg['roadtype'] == 'straight':
                straight += 1

            if seg['angle'] < 0:
                seg['angle'] += 360

            if seg['angle'] > 0:
                angles.append(seg['angle'])

            if seg['pivot_off'] > 0:
                pivot_offs.append(seg['pivot_off'])        

            points[seg_id] = utility.Point(seg['x'], seg['y'])
            path_used.append(seg_id)
            
            if seg['key'] in data['execution']['seg_oob_count'].keys(): #fix is seg in oob
                num_oobs += -1
        if not pivot_offs:
            pivot_offs.append(0)    

        for i in range(0, len(path_used)-1):
            road_distance += utility.get_distance(points[path_used[i]], points[path_used[i+1]])
        # direct_distance = utility.get_distance(Point(data['start'][0], data['start'][1]), Point(data['goal'][0], data['goal'][1]))
        direct_distance = utility.get_distance(points[path_used[0]], points[path_used[-1]])
        # max_distance = data['execution']['maximum_distance']
        # avg_distance = data['execution']['average_distance']

        # if data['execution']['oobs'] > 0 and max_distance >= 3:
        if data['execution']['oobs'] > 0:
            safety = 'unsafe'
        else:
            safety = 'safe'
        
        result = {
            'direct_distance': direct_distance,
            'road_distance': road_distance,
            'num_l_turns': l_turns,
            'num_r_turns': r_turns,
            'num_straights': straight,
            'median_angle': np.median(angles),
            'total_angle': np.sum(angles),
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'max_angle': np.max(angles),
            'min_angle': np.min(angles),
            'median_pivot_off': np.median(pivot_offs),
            'mean_pivot_off': np.mean(pivot_offs),
            'std_pivot_off': np.std(pivot_offs),
            'max_pivot_off': np.max(pivot_offs),
            'min_pivot_off': np.min(pivot_offs),
            'safety': safety
        }
    

        return result
    
    def create_training_test(self, data_set, output_folder):
        df = pd.read_csv(data_set)
        df['safety'] = df['safety'].map(self.label_map)

        is_safe = df['safety'] == 1
        not_safe = df['safety'] == 0

        df_safe = df[is_safe]
        df_not_safe = df[not_safe]
        num_not_safe = int(len(df_not_safe))
        num_safe = int(len(df_safe))
        if num_safe >= num_not_safe:
            num_sample = num_not_safe
            remaining_num = len(df_not_safe) - num_sample
        else:
            num_sample = num_safe
            remaining_num = len(df_safe) - num_sample

        sample_safe = df_safe.sample(n=num_sample, random_state=1)
        sample_unsafe = df_not_safe.sample(n=num_sample, random_state=1)

        training_set = pd.concat([sample_safe, sample_unsafe])

        # if should_normalize:
        #     training_set = self.min_max_normalize(training_set)
        #     test_set = self.min_max_normalize(test_set)
        #     balanced_set = self.min_max_normalize(balanced_set)

        training_set['safety'] = training_set['safety'].map(self.reverse_label_map)
        training_path = '{}/training.csv'.format(output_folder)
     


        training_set.to_csv(training_path, index=False)
       


        # pandas2arff(training_set, training_path_arff, cleanstringdata=True, cleannan=True)
        # pandas2arff(test_set, test_path_arff, cleanstringdata=True, cleannan=True)
        # pandas2arff(balanced_set, test_balanced_path_arff, cleanstringdata=True, cleannan=True)

        return training_path