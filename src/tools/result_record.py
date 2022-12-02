
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from common.config import *

class ResultRecord:
    def __init__(self):
        self.writer = SummaryWriter()

    def RecordTBData(self, name, y, x):
        self.writer.add_scalar(name, y, x)
    
    def FlushTBData(self):
        self.writer.flush()

    def SaveTxtData(self, scoreList):
        np.savetxt(SaveDataPath, np.array(scoreList))
    
    
