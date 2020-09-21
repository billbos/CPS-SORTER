"""Main `cps_sorter` CLI."""
import collections
import json
import os
import sys
import tempfile
from cps_sorter.services.model_evaluator import ModelEvaluator
from cps_sorter.services.weka_helper import WekaHelper
from cps_sorter.services.real_time_experiments import RealTimeExperimentRunner
from cps_sorter.services.performance_testing import split_data, PerformanceTester
from cps_sorter.services.road_transformer import RoadTransformer
from shutil import copyfile
import click

from cps_sorter import __version__


LOGISTIC_MODEL_BUILDING_JAR =  '{}/services/jars/buildLogisticModel.jar'.format(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_OUTPUT =  os.path.dirname(os.path.realpath(__file__))

def version_msg():
    """Return the CPS-SORTER version, location and Python powering it."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = 'CPS-SORTER %(version)s from {} (Python {})'
    return message.format(location, python_version)



@click.group()
def cli():
    pass

@cli.command()
@click.option('-d','--dataset', 'dataset', default='dataset')
@click.option('-i','--input_dir', 'input_dir' )
@click.option('-o','--output_dir', 'output_dir')
def run_model_eval(dataset, input_dir, output_dir):
    model_evaluator = ModelEvaluator(output_dir)
    model_evaluator.evaluate_models(dataset, input_dir)
    print('Finished: {}/{}_results.csv'.format(output_dir, dataset))


@cli.command()
@click.option('-d','--dataset', 'dataset', default='dataset')
@click.option('-i','--input_dir', 'input_dir' )
@click.option('-o','--output_dir', 'output_dir')
@click.option('-r','--rounds', 'rounds')
@click.option('-rt','--ratio', 'ratio')
def run_round_based_eval(dataset, input_dir, output_dir, rounds, ratio):

    # safe_dir = 'D:/MasterThesis/DataSet/performance_test/beamng/safe'
    # unsafe_dir = 'D:/MasterThesis/DataSet/performance_test/beamng/unsafe'
    # output = 'D:/MasterThesis/Results/Performance/Offline'
    road_transformer = RoadTransformer()
    with tempfile.TemporaryDirectory() as temp_dir:
        safe_dir = '{}/safe'.format(temp_dir)
        unsafe_dir = '{}/unsafe'.format(temp_dir)

        for test_file in os.listdir(input_dir):
            file_path = '{}/{}'.format(input_dir, test_file)
            try:
                with open(file_path) as json_file:
                    test = json.load(json_file)
                    if test['execution']['oobs'] > 0:
                        copyfile(file_path, '{}/unsafe/{}'.format(temp_dir.name, test_file))
                    else:
                        copyfile(file_path, '{}/safe/{}'.format(temp_dir.name, test_file))
            except Exception as e:
                print('test file {}: {}'.format(test_file, e))
                continue
        training_dir, test_dir = split_data(safe_dir, unsafe_dir, temp_dir, 0.8, float(ratio))
        
        weka_helper = WekaHelper()
        data_file = '{}/{}'.format(temp_dir, 'data_set.csv')
        data_file = road_transformer.transform_to_training_data(training_dir, data_file, 'default')
        training_file = road_transformer.create_training_test(data_file, temp_dir)
        weka_helper.build_models(trainings_file=training_file, temp_dir=temp_dir, models=['J48.model', 'RandomForest.model', 'Logistic.model'])
        
        tester = PerformanceTester(weka_helper)
        
        result = {}
        tests = ['{}/{}'.format(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
        result['random_fix'] = tester.get_random_baseline_fixed_test_num(test_set=tests, num_tests=10, rounds=rounds)
        result['random_reach'] = tester.get_random_baseline_reach_unsafe_num(test_set=tests, num_unsafe=10, rounds=rounds)
       
        models = weka_helper.get_models()
        result['model_fix'] = tester.model_based_fixed_baseline(tests, num_tests=10, rounds=rounds, models=models)
        result['model_reach'] = tester.get_model_baseline_reach_unsafe_num(tests, num_unsafe=10, rounds=rounds, models=models)
        with open('{}/{}_rounds_tests_ratio_{}.json'.format(output_dir, rounds, ratio), 'w') as outfile:
            outfile.write(json.dumps(result, sort_keys=True, indent=4))
    print('{}/{}_rounds_tests_ratio_{}.json'.format(output_dir, rounds, ratio))
   

@cli.command()
@click.option('-i','--init_data', 'init_data', default=None)
@click.option('-o','--output_dir', 'output_dir', default=DEFAULT_OUTPUT)
@click.option('-t','--time-budget', 'time_budget', default='360')
@click.option('--adaptive/--no-adaptive', default=True)
def run_real_time_eval(init_data, output_dir, time_budget, adaptive):
    temp_dir = tempfile.TemporaryDirectory()
    output_dir = output_dir

    weka = WekaHelper(model_building_jar=LOGISTIC_MODEL_BUILDING_JAR)
    
    test_generator = RealTimeExperimentRunner(temp_dir=temp_dir, weka_helper=weka, output_dir=output_dir)
    result = test_generator.run_experiment(time_budget=int(time_budget), weka_model='Logistic.model', bulk_size=20, init_data=init_data, adaptive=adaptive)
    output_file = test_generator.write_result_to(result, output_dir)
    print(output_file)
 

if __name__ == '__main__':
    init_data = 'C:/workspace/MasterThesis/complete_training.csv'
    cli()
