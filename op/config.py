#!/usr/bin/python
# -*- coding: UTF-8 -*- 

from common import * 
import os
import logging

current_path = os.path.abspath(__file__)
op_path = os.path.dirname(current_path)
root_path = os.path.dirname(op_path)
src_path = root_path + "/src"
logs_path = root_path + "/logs"
models_path = root_path + "/models"

logging_path = logs_path + "/init.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=logging_path, level=logging.DEBUG, format=LOG_FORMAT)

# algorithm = vanila_pg or npg or trpo or ppo(默认)
algorithm = "ppo"

def choose_algorithm():
    logging.info(f"We choose algorithm: {algorithm}")
    sed(r'from algo.* import train_model', f"from algo.{algorithm} import train_model", src_path + "/main_platoon_cooperative_reliability.py")
