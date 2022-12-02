#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os

CurrentPath = os.path.abspath(__file__)
CommonPath = os.path.dirname(CurrentPath)
SrcPath = os.path.dirname(CommonPath)
RootPath = os.path.dirname(SrcPath)
LogsPath = RootPath + "/logs"
ModelsPath = RootPath + "/models"
DataPath = RootPath + "/data"
SaveModelsPath = ModelsPath + "/AgentPlatoon"
SaveDataPath = DataPath + "/scoreList.txt"
SavePDFPath = DataPath + "/ppo.pdf"

LoggingPath = LogsPath + "/trace.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=LoggingPath, level=logging.DEBUG, format=LOG_FORMAT)