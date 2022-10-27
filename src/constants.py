import os
import inspect

__file = inspect.getfile(lambda: None)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file), '..'))
USER_DIR = os.path.dirname(ROOT_DIR) # One level up ROOT_DIR.
NTBK_DIR = os.path.join(ROOT_DIR, 'notebooks')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')