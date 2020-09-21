import datetime
from cps_sorter.services.weka_helper import WekaHelper
from cps_sorter.services.road_transformer import RoadTransformer
from asfault import config, experiments
from asfault.beamer import *
from asfault.network import *
from asfault.evolver import *
from asfault.graphing import *
from asfault.plotter import *
from deap import base, creator, tools
from subprocess import Popen
import tempfile
import csv
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join

DEFAULT_LOG = 'asfault.log'
DEFAULT_ENV = os.path.join(str(Path.home()), '.asfaultenv')

def log_exception(extype, value, trace):
    l.exception('Uncaught exception:', exc_info=(extype, value, trace))


def setup_logging(log_file):
    file_handler = l.FileHandler(log_file, 'a', 'utf-8')
    term_handler = l.StreamHandler()
    l.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                  level=l.INFO, handlers=[term_handler, file_handler])
    sys.excepthook = log_exception
    l.info('Started the logging framework writing to file: %s', log_file)



class RealTimeExperimentRunner:
    def __init__(self, temp_dir=None, weka_helper=None, output_dir=None):
        setup_logging(DEFAULT_LOG)
        ensure_environment(DEFAULT_ENV)
        generate_factories()
        config.ex.ai_controlled = 'true'
        if not temp_dir:
            temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.test_dir = os.mkdir(os.path.join(self.temp_dir.name, 'test_files'))
        self.weka_helper = weka_helper
        self.road_transformer = RoadTransformer()
        self.runner_factory = gen_beamng_runner_factory(config.ex.get_level_dir(), config.ex.host, config.ex.port, plot=False)

    def run_experiment(self, time_budget, weka_model, bulk_size=10, init_data='', adaptive=True):
        results = {
            'generated_tests': 0,
            'tested_files': 0,
            'unsafe_cases': 0,
            'safe_cases': 0,
            'predicated_as_safe': 0,
            'time_test_generation': 0,
            'time_predictions': 0,
            'time_test_run': 0,
            'time_test_evaluation': 0,
            'time_safe_test_run':0,
            'time_unsafe_test_run':0,
            'building_model': 0,
            'init_data_time': 0,
            'parameters': {
                'time_budget': '{} mins'.format(time_budget),
                'weka_model': weka_model,
                'bulk_size': bulk_size,
                'init_data': init_data
            }
        }
        init_start_time = datetime.datetime.now()
        self.log_file = open('{}/log_file.txt'.format(self.output_dir), 'w')
        counter = 0
        rounds = []
        test_factory = RoadTestFactory(config.ev.bounds)
        start_time = datetime.datetime.now()
        self.log_file.write('Start Time: {} \n'.format(start_time))
        print('Start time: {}'.format(start_time))
        if not init_data:
            self.training_file = '{}/trainings_file.csv'.format(self.output_dir)
            init_tests = self.generate_init_data(3*bulk_size, test_factory)
            self.road_transformer.transform_tests_to_training_data(init_tests, self.training_file, True)
            for test in init_tests:
                with open('{}/test_{}.json'.format(self.output_dir, counter), 'w') as out:
                    out.write(json.dumps(RoadTest.to_dict(test), sort_keys=True, indent=4))
                counter += 1
            results['init_data_time'] += (datetime.datetime.now() - init_start_time).total_seconds()
            self.weka_helper.build_models(self.training_file, self.output_dir, [weka_model])
        else:
            self.training_file = init_data
            self.weka_helper.build_models(self.training_file, self.output_dir, [weka_model])

        end_time = datetime.datetime.now() + datetime.timedelta(minutes=time_budget)
        
        while datetime.datetime.now() < end_time:
            round = {
                'num_safe_pred': 0,
                'num_unsafe_pred': 0,
                'num_safe': 0,
                'num_unsafe': 0,
                'false_positive': 0,
                'true_positive': 0,
            }
            new_tests = []
            while len(new_tests) < 20:
                if datetime.datetime.now() > end_time:
                        break
                start_generating = datetime.datetime.now()
                to_predict, test_cases = self.generate_test_cases(test_factory, bulk_size)
                end_time_generating = datetime.datetime.now()
                results['time_test_generation'] += (end_time_generating - start_generating).total_seconds()
                predictions = self.weka_helper.make_bulk_predictions(weka_model, to_predict, bulk_size)
                end_prediction = datetime.datetime.now()
                results['time_predictions'] += (end_prediction - end_time_generating).total_seconds()
                results['generated_tests'] += bulk_size
                for c, prediction in enumerate(predictions):
                    if datetime.datetime.now() > end_time:
                        break
                    if prediction == 'unsafe':
                        results['tested_files'] += 1
                        round['num_unsafe_pred'] += 1
                        test_cases[c].execution = self.run_test(test_cases[c])
                        new_tests.append(test_cases[c])
                        results['time_test_run'] += (test_cases[c].execution.end_time - test_cases[c].execution.start_time).total_seconds()
                        res = self.evaluate_test_case(test_cases[c])
                        if res == 'safe':
                            round['num_safe'] += 1
                            round['false_positive'] += 1
                            results['safe_cases'] += 1
                            results['time_safe_test_run'] += (test_cases[c].execution.end_time - test_cases[c].execution.start_time).total_seconds()
                            self.log_file.write('{}: Mistaken Safe Test Case for Unsafe num: {} \n'.format(datetime.datetime.now(), results['safe_cases']))
                        elif res == 'unsafe':
                            round['num_unsafe'] += 1
                            round['true_positive'] += 1
                            results['unsafe_cases'] += 1
                            results['time_unsafe_test_run'] += (test_cases[c].execution.end_time - test_cases[c].execution.start_time).total_seconds()
                            self.log_file.write('{}: Found Unsafe Test Case num: {} \n'.format(datetime.datetime.now(), results['unsafe_cases']))
                        results['time_test_evaluation'] = (end_prediction - end_time_generating).total_seconds()
                        print('Generated: {} files'.format(results['tested_files']))
                    else:
                        results['predicated_as_safe'] += 1
                        round['num_safe_pred'] += 1
                    end_evaluation = datetime.datetime.now()
                    results['time_test_evaluation'] = (end_evaluation-end_prediction).total_seconds()
                    with open('{}/test_{}.json'.format(self.output_dir, counter), 'w') as out:
                        out.write(json.dumps(RoadTest.to_dict(test_cases[c]), sort_keys=True, indent=4))
                    counter += 1
                    results['time_test_persisting_rejected'] = (datetime.datetime.now()-end_evaluation).total_seconds()
                    if counter % 10 ==0:
                        elapsed_time = datetime.datetime.now() - start_time
                        print("::::::::::::Elapsed Time::::::::: {}".format(str(datetime.timedelta(seconds=elapsed_time.total_seconds()))))
                        print("Tempresult: {}".format(results))
            if adaptive:
                start_building_time = datetime.datetime.now()
                self.road_transformer.transform_tests_to_training_data(new_tests, self.training_file, False)
                self.weka_helper.rebuild_models(self.training_file, self.output_dir, [weka_model])
                results['building_model'] += (datetime.datetime.now() - start_building_time).total_seconds()
            round['unsafe_precision'] =  round['true_positive'] / round['num_unsafe_pred']
            rounds.append(round)
            print("Round: {}".format(round))
            self.log_file.write('{}: Testprediction: {} \n'.format(datetime.datetime.now(), round['true_positive']/round['num_unsafe_pred']))
        results['rounds'] = rounds
        return results



    def generate_init_data(self, bulk_size, factory):
        tests = []
        while len(tests) < bulk_size:
            test = factory.generate_random_test()
            test.execution = self.run_test(test)
            tests.append(test)
        return tests

    # def generate_tests(self, time_budget, weka_model, bulk_size=1):
    #     self.log_file = open('{}/log_file.txt'.format(self.output_dir), 'w')
    #     counter = 0
    #     results = {
    #         'generated_tests': 0,
    #         'tested_files': 0,
    #         'unsafe_cases': 0,
    #         'safe_cases': 0,
    #         'predicated_as_safe': 0,
    #         'time_test_generation': 0,
    #         'time_predictions': 0,
    #         'time_test_run': 0
    #     }
    #     test_factory = RoadTestFactory(config.ev.bounds)

    #     start_time = datetime.datetime.now()
    #     self.log_file.write('Start Time: {} \n'.format(start_time))
    #     end_time = start_time + datetime.timedelta(minutes=time_budget)
    #     print('Start time: {}'.format(start_time))
        
    #     while datetime.datetime.now() < end_time:
    #         remaining_time = end_time - datetime.datetime.now()
    #         if remaining_time.total_seconds() < 3600 and remaining_time.total_seconds() > 1800:
    #             bulk_size = 50
    #         elif remaining_time.total_seconds() < 1800 and remaining_time.total_seconds() > 600:
    #             bulk_size = 20
    #         elif remaining_time.total_seconds() < 600:
    #             bulk_size = 5
    #         start_generating = datetime.datetime.now()
    #         to_predict, test_cases = self.generate_test_cases(test_factory, bulk_size)
    #         end_time_generating = datetime.datetime.now()
    #         results['time_test_generation'] += (end_time_generating - start_generating).total_seconds()
    #         # test_file = tempfile.NamedTemporaryFile(delete=False)
    #         predictions = self.weka_helper.make_bulk_predictions(weka_model, to_predict, bulk_size)
    #         end_prediction = datetime.datetime.now()
    #         results['time_predictions'] += (end_prediction - end_time_generating).total_seconds()
    #         results['generated_tests'] += bulk_size
    #         for c, prediction in enumerate(predictions):
    #             if datetime.datetime.now() > end_time:
    #                 break
    #             if prediction == 'unsafe':
    #                 results['tested_files'] += 1
    #                 test_cases[c].execution = self.run_test(test_cases[c])
    #                 results['time_test_run'] += (test_cases[c].execution.end_time - test_cases[c].execution.start_time).total_seconds()
    #                 res = self.evaluate_test_case(test_cases[c])
    #                 if res == 'safe':
    #                     results['safe_cases'] += 1
    #                     self.log_file.write('{}: Mistaken Safe Test Case for Unsafe num: {} \n'.format(datetime.datetime.now(), results['safe_cases']))
    #                 elif res == 'unsafe':
    #                     results['unsafe_cases'] += 1
    #                 self.log_file.write('{}: Found Unsafe Test Case num: {} \n'.format(datetime.datetime.now(), results['unsafe_cases']))
    #                 print('Generated: {} files'.format(results['tested_files']))
    #             else:
    #                 # results['predicted_as_safe'].append(test_case)
    #                 results['predicated_as_safe'] += 1

    #             with open('{}/test_{}.json'.format(self.output_dir, counter), 'w') as out:
    #                 out.write(json.dumps(RoadTest.to_dict(test_cases[c]), sort_keys=True, indent=4))

    #             counter += 1
    #             if counter % 10 ==0:
    #                 elapsed_time = datetime.datetime.now() - start_time
    #                 # if elapsed_time <
    #                 print("::::::::::::Elapsed Time::::::::: {}".format(str(datetime.timedelta(seconds=elapsed_time.total_seconds()))))
    #                 print("Tempresult: {}".format(results))

    #     self.log_file.write('End time: {}'.format(datetime.datetime.now()))


    #     print('End time: {}'.format(datetime.datetime.now()))
    #     print('Files: {}'.format(results['generated_tests']))
    #     print('test endeded: results: {}'.format(result))
    #     return results

                
    def generate_test_cases(self, factory, bulk_size=1):
        tests = []
        while len(tests) < bulk_size:
            test = factory.generate_random_test()
            tests.append(test)
        
        to_predict = self.road_transformer.convert_to_test_bulk(tests)
            
        return to_predict, tests

   
    def get_distance(self, point_a, point_b):
        return np.sqrt( ((point_a.x-point_b.x)**2)+((point_a.y-point_b.y)**2))

  
    def evaluate_test_case(self, test_case):
        execution = test_case.execution
        if execution.oobs > 0 and execution.reason == 'off_track':
            return 'unsafe'
        else:
            return 'safe'


    def run_test(self, test):
        while True:
            # Test Runner is bound to the test. We need to configure the factory to return a new instance of a runner
            # configured to use the available BeamNG
            runner = self.runner_factory(test)
            try:
                execution = runner.run()
                return execution
            except Exception as e:
                l.error('Error running test %s', test)
                l.exception(e)
                sleep(30.0)
            finally:
                runner.close()

    def write_result_to(self, result, output):
        with open('{}/results.json'.format(output), 'w') as outfile:
            json.dump(result, outfile)
        return '{}/results.json'.format(output)

def milliseconds():
    return round(time() * 1000)


def read_environment(env_dir):
    l.info('Starting with environment from: %s', env_dir)
    config.load_configuration(env_dir)


def ensure_environment(env_dir):
    if not os.path.exists(env_dir):
        l.info('Initialising empty environment: %s', env_dir)
        config.init_configuration(env_dir)
    read_environment(env_dir)


if __name__=='__main__':
    # # Online Testing
    # temp_dir = tempfile.TemporaryDirectory()
    # output_dir = 'C:/Users/bboss/.asfaultenv/output/tests'
    # # output_res = 'C:/Users/bboss/.asfaultenv/output/tests/result.txt '
    # training_dir = 'D:/MasterThesis/Dataset/performance_test/beamng/complete'
    # data_file = '{}/data_file.csv'.format(temp_dir.name)
    # road_tr = RoadTransformer()
    # road_tr.transform_to_training_data(training_dir, data_file, 'beamng')
    # trainings_file = road_tr.create_training_test(data_file, temp_dir.name)

    # weka = WekaHelper()
    # weka.build_models(trainings_file=trainings_file, temp_dir=temp_dir.name, models=['Logistic.model'])
       
    # test_generator = Testgenerator(temp_dir=temp_dir, weka_helper=weka, output_dir=output_dir)
    # result = test_generator.generate_tests(time_budget=360, weka_model='Logistic.model', bulk_size=200)
    # test_generator.write_result_to(result, output_dir)



    temp_dir = tempfile.TemporaryDirectory()
    output_dir = 'C:/Users/bboss/.asfaultenv/output/tests'
    init_data = 'C:/Users/bboss/.asfaultenv/output/tests/trainings_file.csv'
    # data_file = '{}/data_file.csv'.format(temp_dir.name)
    LOGISTIC_MODEL_BUILDING_JAR = 'C:/workspace/MasterThesis/scripts/jars/buildLogisticModel.jar'

    road_tr = RoadTransformer()
    weka = WekaHelper(model_building_jar=LOGISTIC_MODEL_BUILDING_JAR)
    
    test_generator = RealTimeExperimentRunner(temp_dir=temp_dir, weka_helper=weka, output_dir=output_dir)
    result = test_generator.run_experiment(time_budget=60, weka_model='Logistic.model', bulk_size=5)
    output_file = test_generator.write_result_to(result, output_dir)
    print(output_file)
 