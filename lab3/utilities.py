import os
import json


class Constants:

    def __init__(self):
        self.ROOT = ''
        self.DATA_ROOT = 'data'
        self.LOG_ROOT = 'logs'
        self.MODELS_ROOT = 'models'

        self.NUM_CLASSES = 61
        
        self.INPUT_KIND = 'dynamic_lmfcc'
        self.EPOCHS = 1
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 256
        self.HIDDEN_LAYERS = 4

    def net_params_to_dictionary(self):
        return {
            'INPUT_KIND': self.INPUT_KIND,
            'EPOCHS': self.EPOCHS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'BATCH_SIZE': self.BATCH_SIZE,
            'HIDDEN_LAYERS': self.HIDDEN_LAYERS
        }


class Logging:

    def __init__(self, log_name='log.json'):
        self.co = Constants()
        self.LOG_ROOT = self.co.LOG_ROOT
        self.LOG_NAME = os.path.join(self.LOG_ROOT, log_name)

        self._log_dir_creation(self.LOG_ROOT)
        self._log_creation(self.LOG_NAME)

    def _log_dir_creation(self, LOG_ROOT):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

    def _log_creation(self, LOG_NAME):
        if not os.path.isfile(LOG_NAME):
            with open(LOG_NAME, "w") as f:
                log_feed = []
                json_p = json.dumps(log_feed)
                f.write(json_p)

    def store_log_entry(self, entry):
        if not type(entry) == dict:
            raise ValueError('Store dictionary entries, not ',
                             type(entry))

        with open(self.LOG_NAME, "r") as f:
            log_feed = json.load(f)
            log_feed.append(entry)

        with open(self.LOG_NAME, "w") as f:
            json_p = json.dumps(log_feed, indent=3)
            f.write(json_p)

    def read_log(self):
        with open(self.LOG_NAME, 'r') as f:
            log_feed = json.load(f)
            # If the above line does not work, comment and use the below
            # log_feed = json.loads(f)

        return log_feed
