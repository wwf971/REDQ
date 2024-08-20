import sqlite3
from typing import TYPE_CHECKING

import _utils_file
def CreateDataBase(FilePath):
    FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    if _utils_file.file_exist(FilePath):
        _utils_file.remove_file(FilePath)
    con = sqlite3.connect(FilePath)
    cursor = con.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS TrainData(
            Loss REAL,
            SampleNum INTEGER,
            SampleNumCorrect INTEGER,
            SampleNumCorrectTop5 INTEGER,
            EpochIndex INTEGER,
            BatchIndex INTEGER,
            BatchIndexTotal INTEGER PRIMARY KEY
        )
        '''
    ) # 最后1项结束后在)前不能有逗号,
    con.commit()
    return con

def InsertIntoDataBase(
    con,
    Loss,
    SampleNum,
    SampleNumCorrect, # SampleCorrectTop1
    SampleNumCorrectTop5,
    EpochIndex, BatchIndex, BatchIndexTotal,
):
    if TYPE_CHECKING:
        con = sqlite3.connect(DataBaseFilePath)
    cursor = con.cursor()
    cursor.execute('''
        INSERT INTO TrainData (
            Loss, SampleNum, SampleNumCorrect, SampleNumCorrectTop5,
            EpochIndex, BatchIndex, BatchIndexTotal
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            Loss, SampleNum, SampleNumCorrect, SampleNumCorrectTop5,
            EpochIndex, BatchIndex, BatchIndexTotal
        )
    )
    con.commit()
    return