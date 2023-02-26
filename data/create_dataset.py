from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


TRAIN_START_DATE = '2009-04-01'
TRAIN_END_DATE = '2020-12-31'
VAL_START_DATE = '2021-01-01'
VAL_END_DATE = '2021-12-31'
TEST_START_DATE = '2022-01-01'
TEST_END_DATE = '2022-12-31'

ticker_list = ['QQQ']

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = ticker_list).fetch_data()


train = df[(df.date <= TRAIN_END_DATE)].drop(columns=['day'])
val = df[(df.date >= VAL_START_DATE) & (df.date <= VAL_END_DATE)].drop(columns=['day'])
test = df[(df.date >= TEST_START_DATE)].drop(columns=['day'])

train.to_csv(ticker_list[0]+'_train.csv', header=True, index=False)
val.to_csv(ticker_list[0]+'_val.csv', header=True, index=False)
test.to_csv(ticker_list[0]+'_test.csv', header=True, index=False)