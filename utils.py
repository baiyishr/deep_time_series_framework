
data_source_dictionary = {
    'csv': ['datamodules.csvdatamodule', 'CsvDataModule'],
    'hive': ['datamodules.hivedatamodule', 'HiveDataModule'],
    'sql': ['datamodules.sqldatamodule', 'SqlDataModule'],
    's3': ['datamodules.s3datamodule', 'S3DataModule']
}

model_dictionary = {
    'custom_rnn': ['models.custom_rnn', 'CustomRNN'],
    'customrnn': ['models.custom_rnn', 'CustomRNN'],
    'tst': ['models.tst', 'TSTModel'],
    'TST': ['models.tst', 'TSTModel'],
}

loss_dictionary = {
    'mse': ['loss.loss', 'MSE'],
    'MSE': ['loss.loss', 'MSE'],
    'cross_entropy': ['loss.loss', 'CrossEntropy'],
}

dataset_map={
    "rnn" : ('datamodules.rnndataset', 'RNNDataset'),
    "tst" : ('datamodules.tstdataset', 'TSTDataset')
}

