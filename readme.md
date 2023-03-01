
## Deep Time Series Framework For stock price prediction- Pytorch Lightning

Deep time series framework based on Pytorch Lightning

The purpose of the project is to build a scalable framework to quickly test different variations of transformer-based time-series models for stock price prediction. 


## Todo

- [x] build the skeleton of the framework with datamodule, model, loss functions and callbacks
- [x] add CsvDataModule for loading train/val/test data into DataLoaders
- [x] implement and test the custom_rnn model
- [x] implement and test a Deep transformer model based on this paper: Wu, N., Green, B., Ben, X., & O'Banion, S. (2020). Deep transformer models for time series forecasting: The influenza prevalence case. arXiv preprint arXiv:2001.08317
- [ ] add return prediction and classification
- [ ] implement and test Google's Temporal Fusion Transform model based on this <a href="https://arxiv.org/pdf/1912.09363.pdf">paper</a>
- [ ] Hive, S3 and SqlDataModules are not tested
