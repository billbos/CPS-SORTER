import subprocess
import re
import tempfile
from shutil import copyfile
import os

DEFAULT_PREDICTION_JAR = 'C:/workspace/MasterThesis/scripts/jars/makePrediction.jar'
DEFAULT_PREDICTION_BULK_JAR = 'C:/workspace/MasterThesis/scripts/jars/makeBatchPrediction.jar'
DEFAULT_MODEL_BUILDING_JAR = 'C:/workspace/MasterThesis/scripts/jars/train_models.jar'
DEFAULT_MODEL_EVALUATOR_JAR = 'C:/workspace/MasterThesis/scripts/jars/ModelEvaluator.jar'

class WekaHelper:
    def __init__(self, prediction_jar=DEFAULT_PREDICTION_JAR, model_building_jar=DEFAULT_MODEL_BUILDING_JAR, temp_dir=None, model_evaluator_jar=DEFAULT_MODEL_EVALUATOR_JAR):
        self.prediction_jar = prediction_jar
        self.bulk_prediction_jar = DEFAULT_PREDICTION_BULK_JAR
        self.model_building_jar = model_building_jar
        self.model_evaluator_jar = model_evaluator_jar
        self.models = {}
        if not temp_dir:
            temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir.name
    
    def build_models(self, trainings_file, temp_dir, models=['J48.model', 'RandomForest.model', 'Logistic.model']):
        subprocess.call(['java', '-jar', self.model_building_jar, trainings_file, temp_dir])
        for model in models:
            self.models[model] = '{}/{}'.format(temp_dir, model)

    def rebuild_models(self, trainings_file, temp_dir, models=['J48.model', 'RandomForest.model', 'Logistic.model']):
        for model in models:
            try:
                os.remove(self.models[model])
            except Exception as e:
                print(e)
        subprocess.call(['java', '-jar', self.model_building_jar, trainings_file, temp_dir])
        for model in models:
            self.models[model] = '{}/{}'.format(temp_dir, model)

    def make_prediction(self, model, to_test):
        try:
            csv_name = '{}.csv'.format(to_test)
            csv_file = copyfile(to_test, csv_name)
            process = subprocess.Popen(['java', '-jar', self.prediction_jar, csv_file, self.models[model]], stdout=subprocess.PIPE)
            safe_pattern = '.*0.0'
            unsafe_pattern = '.*1.0'
            output = process.stdout.readline()
            output = str(output.strip())
            if re.search(safe_pattern, output):
                result = 'safe'
            elif re.search(unsafe_pattern, output):
                result = 'unsafe'
            print('Result: {}'.format(output))
            return result
        except Exception as e:
            print(e)


    def make_bulk_predictions(self, model, to_test, bulk_size=1):
        try:
            predictions = []
            csv_name = '{}.csv'.format(to_test)
            csv_file = copyfile(to_test, csv_name)
            process = subprocess.Popen(['java', '-jar', self.bulk_prediction_jar, csv_file, self.models[model]], stdout=subprocess.PIPE)
            for i in range(0, bulk_size):
                safe_pattern = '.*Index {} Result 0.0'.format(i)
                unsafe_pattern = '.*Index {} Result 1.0'.format(i)
                output = process.stdout.readline()
                output = str(output.strip())
                if re.search(safe_pattern, output):
                    predictions.append('safe')
                elif re.search(unsafe_pattern, output):
                    predictions.append('unsafe')
                print('Result: {}'.format(output))
            return predictions

        except Exception as e:
            print(e)

    def evaluate_models(self, dataset_name, trainings_set, test_set, output_file):
        try:
            process = subprocess.Popen(['java', '-jar', self.model_evaluator_jar, trainings_set, test_set, dataset_name, output_file])
        except Exception as e:
            print('WekaHelper:{}'.format(e))

    def get_models(self):
        return self.models