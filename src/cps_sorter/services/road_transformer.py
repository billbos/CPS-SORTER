import numpy as np
import pandas as pd
import cps_sorter.services.utility as utility
import tempfile
import csv
import json
import os
from asfault.tests import RoadTest
import random
from scipy.spatial.distance import directed_hausdorff


class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

    def __eq__(self, other):
        if (self.x == other.x) and (self.y == other.y):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
class RoadTransformer:

    label_map = {
        'unsafe': 0,
        'safe': 1
    }
    reverse_label_map = {
        0: 'unsafe',
        1: 'safe'
    }

    def get_distance(self, point_a, point_b):
        return np.sqrt( ((point_a.x-point_b.x)**2)+((point_a.y-point_b.y)**2))

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

    def extract_segment_features_rows(self, data):
            rows = []
            num_oobs = data['execution']['oobs']
            path = data['path']
            stop_at_last_oob = True
            if data['execution']['reason'] == 'goal_reached':
                stop_at_last_oob = False

            for i in range(0, len(path)):
            
                row = {
                    'is_start_seg': 0,
                    'is_last_seg': 0,
                }
                seg_id = path[i]
                seg = data['network']['nodes'][str(seg_id)]
                prev_seg = next_seg = None
                if i == 0:
                    row['is_start_seg'] = 1
                else:
                    prev_seg = data['network']['nodes'][str(path[i-1])]
                if i == len(path)-1:
                    row['is_last_seg'] = 1
                else:
                    next_seg = data['network']['nodes'][str(path[i+1])]
                row.update(self.segment_to_feature(seg, prev_seg, next_seg))
                if seg['key'] in data['execution']['seg_oob_count'].keys(): #fix is seg in oob
                    row['safety'] = 'unsafe'
                    num_oobs += -1
                else:
                    row['safety'] = 'safe'
                rows.append(row)
                if num_oobs == 0 and stop_at_last_oob:
                    break
            return rows
    def segment_to_feature(self,segment, prev_seg={}, next_seg={}):
        prev_seg_feature = next_seg_feature = {}
        seg_l_lane =  np.array([(p[0], p[1]) for p in segment['l_lanes'][0]['l_edge']])
        if prev_seg:
            prev_seg_feature = self.segment_extract_feature(prev_seg, 'prev')
            prev_l_lane = np.array([(p[0], p[1]) for p in prev_seg['l_lanes'][0]['l_edge']])
            prev_seg_feature['prev_directed_hausdorff'] = directed_hausdorff(seg_l_lane, prev_l_lane)[0]

        else:
            prev_seg_feature = {
            'prev_angles': -1,
            'prev_is_right_turn': 0,
            'prev_is_left_turn': 0,
            'prev_is_straight': 0,
            'prev_pivot_off': -1,
            'prev_actual_length': -1,
            'prev_direct_length': -1,
            'prev_directed_hausdorff': -1
        }
        if next_seg:
            next_seg_feature = self.segment_extract_feature(next_seg, 'next')
            next_seg_l_lane =  np.array([(p[0], p[1]) for p in next_seg['l_lanes'][0]['l_edge']])
            next_seg_feature['next_directed_hausdorff'] = directed_hausdorff(seg_l_lane, next_seg_l_lane)[0]


        else:
            next_seg_feature = {
            'next_angles': -1,
            'next_is_right_turn': 0,
            'next_is_left_turn': 0,
            'next_is_straight': 0,
            'next_pivot_off': -1,
            'next_actual_length': -1,
            'next_direct_length': -1,
            'next_directed_hausdorff': -1

        }
        seg_feature = self.segment_extract_feature(segment, 'seg')
        result = {**seg_feature, **prev_seg_feature, **next_seg_feature}
        return result

    def segment_extract_feature(self, segment, prefix):
        l_lane = [Point(p[0], p[1]) for p in segment['l_lanes'][0]['l_edge']]
        r_lane = [Point(point[0], point[1]) for point in segment['r_lanes'][0]['r_edge']]

        l_lane_distance = sum([self.get_distance(x,y) for x,y in zip(l_lane[:-1], l_lane[1:])]) 
        r_lane_distance = sum([self.get_distance(x,y) for x,y in zip(r_lane[:-1], r_lane[1:])])
        direct_distance = (self.get_distance(l_lane[0], l_lane[-1]) + self.get_distance(r_lane[0], r_lane[-1])) / 2
        is_right_turn = is_left_turn = is_straight = 0
        if segment['roadtype'] == 'l_turn':
            is_left_turn = 1
        elif segment['roadtype'] == 'r_turn':
            is_right_turn = 1
        elif segment['roadtype'] == 'straight':
            is_straight = 1

        segment_feature = {
            '{}_angles'.format(prefix): segment['angle'],
            '{}_is_right_turn'.format(prefix): is_right_turn,
            '{}_is_left_turn'.format(prefix): is_left_turn,
            '{}_is_straight'.format(prefix): is_straight,
            '{}_pivot_off'.format(prefix): segment['pivot_off'],
            '{}_actual_length'.format(prefix): (r_lane_distance+l_lane_distance)/2,
            '{}_direct_length'.format(prefix): direct_distance
        }
        return segment_feature