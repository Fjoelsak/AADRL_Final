import numpy as np 
import pandas as pd 
from collections import defaultdict

class Preprocessor():
    """
    This class does the preprocessing for the financial market data, including 
    the calculation of technical indicators like MACD, RSI, etc., and merges 
    this data with the results of sentiment analysis. The combined dataframe 
    serves as the foundation for the agent-environment interaction.

    Note that the calculation of the sentiments will not be part of this
    code, since it takes a very long time to generate sentiments. (The
    code for the sentiment analysis can be found in sentiment_analysis.ipynb). 
    Therefore, only the results of the sentiment analysis will be used in this
    class.
    """    
    DATE = 'date'
    STOCK = 'stock'
    SENTIMENT = 'sentiment'
    CLOSE_COLUMN = 'Close'
    HIGH_COLUMN = 'High'
    OPEN_COLUMN = 'Open'
    LOW_COLUMN = 'Low'
    
    def __init__(
            self, 
            financial_data : pd.DataFrame,
            sentiments: pd.DataFrame,
            ndays_rsi: int = 14,
            ndays_cci: int = 20,
            ndays_adx: int = 14, 
            ndays_macd: tuple = (26,12),
            columns_to_keep: list = ['Adj Close', 
                                     'MACD', 
                                     'RSI', 
                                     'ADX', 
                                     'CCI', 
                                     'sentiment']
            ):
        """
        Initialize the Preprocessor

        Params:
            - financial_data (pd.DataFrame): financial market data containing
              for each ticker and date the corresponding OHLC-data 
              
              Structure of the dataframe (with illustrative numbers):
                Price	        Adj Close	        ...	      Volume
                Ticker	    NVDA	    NFLX	    ...	    NVDA	    NFLX	
                Date																					
                2009-01-02	1.997425	4.267143	...	   1228312     50345343
                2009-01-05	2.034118	4.562857	...	   2504923     92841924
                2009-01-06	2.102915	4.705714	...	   3124935     50324042


            - sentiments (pd.DataFrame): datframe containing for each ticker 
              the sentiment for the different dates. This table should be 
              generated as part of the sentiment analysis in a different step.

              Structure of the dataframe:
                date	    stock	sentiment
              0	2009-06-16	M	     0
              1	2009-06-23	KO	     0
              2	2009-07-27	MRK	     0
              3	2009-08-06	M	    -1
              Sentiment 0: neutral, -1: negative, 1: positive

            - ndays_rsi (int)/ndays_cci (int)/ndays_adx (int): time window 
            over which the RSI, CCI and ADX will be calculated

            - ndays_macd (tuple): tuple contains the time window over which 
            the exponential moving average gets calculated. The first entry
            corresponds to the moving average of the longer EMA, while 
            the second refers to the shorter EMA.

            - columns_to_keep (list): columns that one wants to keep for 
            the environment. (default: ['Adj Close','MACD','RSI','ADX', 
            'CCI', 'sentiment'] - these columns are except of the sentiment 
            column the ones mentioned in the paper "DRL for Automated Stock 
            Trading: An Ensemble Strategy")
        """
        # Get the financial data and sentiments
        self.financial_data = financial_data
        self.sentiments = sentiments

        # Get the time windows for the different technical indicators
        self.ndays_rsi = ndays_rsi
        self.ndays_cci = ndays_cci
        self.ndays_adx = ndays_adx
        self.ndays_macd = ndays_macd

        # Define the columns to keep
        self.columns_to_keep = columns_to_keep

        # longest time window to calculate technical 
        # (note that ADX need 2n datapoints for the first ADX value
        # where n is the time window specified in the method)
        self.max_ndays_ti = max(
                                self.ndays_rsi, 
                                self.ndays_cci,
                                self.ndays_adx * 2,
                                self.ndays_macd[0]
                                )

    def get_preprocessed_data(self):
        """
        This method will get the financial market and sentiment analysis 
        data merge them and calculate the technical indicators for each ticker. 
        The result of this method will be the basis for the agent-environment 
        interaction.

        Params:
            - None
        Returns:
            - pd.Dataframe containing the processed data
        """
        # Reshape the sentiment data 
        self._reshape_sentimentdata()

        # Merge the sentiment data and market data 
        self.merged_datasets = self._merge_datasets()
        
        # Note that this table starts at earliest sentiment headline date - 
        # max window to calculate the technical indicator. In this way, when
        # we have the first sentiment, we also have the first values for the
        # technical indicators. 

        # Add the technical indicators 
        self.add_MACD()
        self.add_RSI()
        self.add_ADX()
        self.add_CCI()

        # Keep only the columns that were specified
        filter = self.merged_datasets.columns[
                    self.merged_datasets.columns.get_level_values(0).isin(
                                                        self.columns_to_keep
                                                        )
                    ]
        
        self.merged_datasets = self.merged_datasets[filter]

        # Reorder the columns
        self.merged_datasets = self.merged_datasets.reindex(
            columns=self.merged_datasets.columns.reindex(self.columns_to_keep, 
                                                         level=0)[0]
                                                         )


        # Start the data from the first sentiment onwards. (Remove the extra 
        # n datapoints prior to earliest sentiment headline that were needed
        # to calculate the technical indicators)
        return self.merged_datasets.iloc[self.max_ndays_ti:]

    def _reshape_sentimentdata(self):
        """ 
        This method will reshape the initial sentiment dataset from the
        sentiment analysis such that it matches the structure of the 
        financial market data.

        For example, the data that resulted from the sentiment analysis might
        look like this: 
                date	    stock	sentiment
              0	2009-06-16	M	     0
              1	2009-06-23	KO	     0
              2	2009-07-27	MRK	     0
              3	2009-08-06	M	    -1
        This method will then reshape this table into a dataframe of the 
        following structure:
                        sentiment
                        M      KO   MRK    M
            date        
            2009-06-16	0	   NaN  NaN   NaN
            2009-06-23	NaN	   0    NaN   NaN
            2009-07-27	NaN	   NaN   0    NaN
            2009-08-06	NaN	   NaN  NaN    0

        Params:
            - None
        Returns:
            - None
        """
        # pivot the DataFrame
        self.sentiments = self.sentiments.pivot(
                                index=self.DATE, 
                                columns=self.STOCK, 
                                values=self.SENTIMENT
                                )

        # Create MultiIndex for columns
        self.sentiments.columns = pd.MultiIndex.from_tuples(
                                    [(self.SENTIMENT, col) 
                                    for col in self.sentiments.columns]
                                  )

    def _merge_datasets(self) -> pd.DataFrame:
        """ 
        This method merges the two datasets (one from sentiment analysis and
        financial market data). If for any given day, a stock does not contain 
        a financial news headline, the sentiment will be set as 0.

        Params:
            - None
        Returns:
            - pd.Dataframe: merged dataframe
            Shape:
                        Adj Close	            ...	sentiment
                        EBAY	NFL	            ...	NFLX	NVDA	
            2009-06-15	14.400828	20.931566	...	0.0	    0.0	
            2009-06-16	14.095044	20.710451	...	0.0	    0.0	
            2009-06-17	14.291208	20.253494	...	0.0	    0.0	

        """
        # Convert indeces to datetime
        self.sentiments.index = pd.to_datetime(self.sentiments.index)
        self.financial_data.index = pd.to_datetime(self.financial_data.index)

        # Concat both dataframes
        data_concatted = pd.concat(
                    [self.financial_data, self.sentiments], 
                    axis = 1
                    )
        
        # The market data might have more data than the sentiment analysis 
        # dataframe. Keep only the same dates. To calculate the technical 
        # indicators, we start the market data n-days prior to the minimum date 
        # of the sentiment analysis dataframe. In this way, on the first date 
        # of the sentiment analysis, we have the prices as well as technical 
        # indicators and sentiments.

        # Index where the financial data should start
        start_position = max(0, 
                             data_concatted.index.get_loc(self.sentiments.index.min()) 
                             - self.max_ndays_ti)

        # Get the index of the last sentiment
        end_position = data_concatted.index.get_loc(self.sentiments.index.max())

        # Get the data in between start and end index
        data_concatted = data_concatted.iloc[start_position:end_position + 1]

        # For each date and stock where no financial news headline is available, 
        # the data contains NaN. We will in this case replace it with 0
        data_concatted[self.SENTIMENT] = data_concatted[self.SENTIMENT].fillna(0)

        return data_concatted

    def add_MACD(self):
        """ 
        This method will add to the merged dataframe (containing market data and
        sentiments for different dates), for each ticker the MACD time series. Please
        note that this method will modifiy the merget dataframe. Also note that the
        MACD will be calculated regarding the close prices. 

        Params:
            - None
        Returns:
            - None 
        """
        # Define a default dict to store the MACD time series
        macd_dict = defaultdict()

        for ticker in self.merged_datasets[self.CLOSE_COLUMN].columns:
            # Get the close prices and calculate the MACD
            close_prices_ticker = self.merged_datasets[self.CLOSE_COLUMN][ticker]
            macd = self.calculate_MACD(
                                    pd.Series(close_prices_ticker.values),
                                    short_span=self.ndays_macd[1],
                                    long_span=self.ndays_macd[0]
                                    )
            
            # Save the time series
            macd_dict[('MACD',ticker)] = macd
        
        # Convert the dictionary to a dataframe and change the indeces to dates
        df = pd.DataFrame(macd_dict)
        df.index = self.merged_datasets[self.CLOSE_COLUMN].index

        # Merge the initial dataset with the MACD time series
        self.merged_datasets = pd.concat([self.merged_datasets, df], axis=1)

    def add_RSI(self):
        """ 
        This method will add to the merged dataframe (containing market data and
        sentiments for different dates), for each ticker the RSI time series. Please
        note that this method will modifiy the merget dataframe. Also note that the
        RSI will be calculated regarding the close prices. 

        Params:
            - None
        Returns:
            - None 
        """
        # Define a default dict to store the RSI time series
        rsi_dict = defaultdict()

        for ticker in self.merged_datasets[self.CLOSE_COLUMN].columns:
            # Get the close prices and calculate the MACD
            close_prices_ticker = self.merged_datasets[self.CLOSE_COLUMN][ticker]
            macd = self.calculate_RSI(
                                      pd.Series(close_prices_ticker.values), 
                                      span = self.ndays_rsi
                                      )
            
            # Save the time series
            rsi_dict[('RSI',ticker)] = macd
        
        # Convert the dictionary to a dataframe and change the indeces to dates
        df = pd.DataFrame(rsi_dict)
        df.index = self.merged_datasets[self.CLOSE_COLUMN].index

        # Merge the initial dataset with the MACD time series
        self.merged_datasets = pd.concat([self.merged_datasets, df], axis=1)
    
    def add_ADX(self):
        """
        This method will add to the merged dataframe (containing market data and
        sentiments for different dates), for each ticker the ADX time series. Please
        note that this method will modifiy the merget dataframe. 
        """
        # Define a default dict to store the ADX time series
        adx_dict = defaultdict()

        for ticker in self.merged_datasets[self.CLOSE_COLUMN].columns:
            # Get the close prices and calculate the MACD
            close_prices_ticker = self.merged_datasets[self.CLOSE_COLUMN][ticker]
            high_prices_ticker = self.merged_datasets[self.HIGH_COLUMN][ticker]
            low_prices_ticker = self.merged_datasets[self.LOW_COLUMN][ticker]

            adx = self.calculate_ADX(
                                      pd.Series(high_prices_ticker.values), 
                                      pd.Series(close_prices_ticker.values), 
                                      pd.Series(low_prices_ticker.values),
                                      n = self.ndays_adx
                                      )
            
            # Save the time series
            adx_dict[('ADX',ticker)] = adx
        
        # Convert the dictionary to a dataframe and change the indeces to dates
        df = pd.DataFrame(adx_dict)
        df.index = self.merged_datasets[self.CLOSE_COLUMN].index

        # Merge the initial dataset with the MACD time series
        self.merged_datasets = pd.concat([self.merged_datasets, df], axis=1)
    
    def add_CCI(self):
        """
        This method will add to the merged dataframe (containing market data and
        sentiments for different dates), for each ticker the CCI time series. Please
        note that this method will modifiy the merget dataframe. 
        """
        # Define a default dict to store the CCI time series
        cci_dict = defaultdict()

        for ticker in self.merged_datasets[self.CLOSE_COLUMN].columns:
            # Get the close prices and calculate the MACD
            close_prices_ticker = self.merged_datasets[self.CLOSE_COLUMN][ticker]
            high_prices_ticker = self.merged_datasets[self.HIGH_COLUMN][ticker]
            low_prices_ticker = self.merged_datasets[self.LOW_COLUMN][ticker]

            cci = self.calculate_CCI(
                                      pd.Series(high_prices_ticker.values), 
                                      pd.Series(low_prices_ticker.values),
                                      pd.Series(close_prices_ticker.values), 
                                      ndays = self.ndays_cci
                                      )
            
            # Save the time series
            cci_dict[('CCI',ticker)] = cci
        
        # Convert the dictionary to a dataframe and change the indeces to dates
        df = pd.DataFrame(cci_dict)
        df.index = self.merged_datasets[self.CLOSE_COLUMN].index

        # Merge the initial dataset with the MACD time series
        self.merged_datasets = pd.concat([self.merged_datasets, df], axis=1)

    @staticmethod
    def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
        """
        Calculate the Exponential Moving Average (EMA) for a given series (as 
        specified in "Technical Analysis: Power Tools for Active Investors" by 
        Appel pp. 134-136 
        
        Params:
            - prices (pd.Series): The time series data for which one wants to 
              calculate the EMA
            - span (int): n-day span over which the exponential moving average 
              should calculated. Since the first EMA-value will be calculated on 
              the basis of the n-day Simple Moving Average (SMA), the span will 
              have an effect on the n-day SMA.
        Returns:
            - pd.Series: The EMA values.
        """
        # Create empty pandas series to save the values later
        ema = pd.Series(np.nan, prices.index)

        # Calculate the n-day moving average (initial value for ema calculation) 
        ma = prices.iloc[:span].mean()  
        multiplier = 2 / (span + 1)

        # Loop over the prices and calculate the ema value
        for idx,price in enumerate(prices.iloc[span:]):            
            ma = multiplier * price + (1-multiplier)*ma
            
            # Save the value
            ema.iloc[idx+span] = ma

        return ema 

    @staticmethod
    def calculate_MACD(
                    prices: pd.Series, 
                    short_span: int = 12, 
                    long_span: int = 26
                    ):
        """
        Calculate the MACD for a given time series (see in "Technical Analysis: 
        Power Tools for Active Investors" by Appel pp. 134-136,167 for further
        notes on what the MACD is)

        Params:
            - df (pd.DataFrame): DataFrame containing the stock data with a 
              'Close' column.
            - short_span (int): span of the EMA with the shorter time window.
              Default: 12 (as specified in "Technical Analysis: Power Tools 
              for Active Investors")
            - long_span (int): span of the EMA with the bigger time window.
              Default: 26 (as defined in "Technical Analysis: Power Tools 
              for Active Investors")
        Returns:
            - pd.DataFrame: DataFrame with MACD and Signal line added as columns.
        """
        # Calculate the 12-period EMA
        short_ema = Preprocessor.calculate_ema(
                                    prices, 
                                    span = short_span
                                    )
        
        # Calculate the 26-period EMA
        long_ema = Preprocessor.calculate_ema(
                                    prices,
                                    span = long_span
                                    )
        
        # Calculate the MACD 
        return short_ema - long_ema

    @staticmethod
    def calculate_RSI(
                    close_prices: pd.Series, 
                    span:int = 14
                    ):
        """
        Calculate for a given time series the Relative Strength Index (RSI). 
        This method follows the instructions mentioned in "New Concepts in 
        Technical Trading Systems" by Welles Wilder (section VI).
        
        Params:
            - close_prices (pd.Series): The time series data for which one wants 
            to calculate the RSI
            - span (int): number of days over which the relative strength 
              will be calculated. Default: 14, as defined by Wilder
        """
        # Calculate the price differences
        deltas = close_prices.diff().iloc[1:].reset_index(drop=True)

        # Average increase and decreae over n-days
        avg_up, avg_down = (deltas[:span].clip(lower=0).mean(), 
                            -deltas[:span].clip(upper=0).mean())
        
        rsi_series = pd.Series(np.nan, close_prices.index)

        # Set the first RSI value
        rsi_series[span] = 100 if avg_down == 0 else \
                             100 - (100 / (1 + avg_up / avg_down))

        # Loop over the different deltas 
        for idx, d in enumerate(deltas[span:]):
            # Calculate new average upwards and downwards
            avg_up = (avg_up * (span-1) + np.maximum(d, 0)) / span 
            avg_down = (avg_down * (span-1) - np.minimum(d, 0)) / span

            # Calculate and set new RSI value 
            rsi_series.iloc[idx + span + 1] = 100 if avg_down == 0 else \
                                    100 - (100 / (1 + avg_up / avg_down))
        return rsi_series

    @staticmethod
    def calculate_ADX(
                    high_prices: pd.Series,
                    close_prices: pd.Series,
                    low_prices: pd.Series,
                    n: int = 14):
        """
        Calculate the Average Directional Index (ADX) for the given data.
        This method follows the instructions mentioned in "New Concepts in 
        Technical Trading Systems" by Welles Wilder.

        Params:
            - data (pandas.DataFrame): DataFrame containing 'High', 'Low', 
              and 'Close' price columns.
            - n (int): Period for calculating ADX, default is 14.

        Returns:
            - pandas.DataFrame: DataFrame with the ADX and related indicators.
        """        
        # Calculate true range
        true_range = np.maximum(
            abs(high_prices - low_prices),
            np.maximum(abs(high_prices - close_prices.shift(1)), 
                       abs(low_prices - close_prices.shift(1)))
        )

        # Calculate directional movement 
        DM_plus_nonsmoothed = np.where((high_prices - high_prices.shift(1)) > 
                           (low_prices.shift(1) - low_prices),
                            np.maximum(high_prices - high_prices.shift(1), 0), 0)
        DM_minus_nonsmoothed = np.where((low_prices.shift(1) - low_prices) > 
                            (high_prices - high_prices.shift(1)),
                            np.maximum(low_prices.shift(1) - low_prices, 0), 0)


        # Initialize variables
        TR_n = true_range[:n].sum()
        DM_plus = DM_plus_nonsmoothed[:n].sum()
        DM_minus = DM_minus_nonsmoothed[:n].sum()
        
        smoothed_data = np.full(
                            shape = (high_prices.index.stop,3), 
                            fill_value=np.nan)
        
        for idx, d in enumerate(true_range[n:]): 
            # Smooth the values
            TR_n = TR_n - TR_n/n + true_range[idx+n]
            DM_plus = DM_plus - DM_plus/n + DM_plus_nonsmoothed[idx+n]
            DM_minus = DM_minus - DM_minus/n + DM_minus_nonsmoothed[idx+n]

            # Save the values
            smoothed_data[idx+n,0] = TR_n
            smoothed_data[idx+n,1] = DM_plus
            smoothed_data[idx+n,2] = DM_minus

        # Calculate plus and minus DI and DX
        di_plus = 100 * (smoothed_data[:,1]  / smoothed_data[:,0])
        di_minus = 100 * (smoothed_data[:,2]  / smoothed_data[:,0])
        dx = 100 * (abs(di_plus - di_minus)) / abs(di_plus + di_minus)
        
        # Calculate initial ADX 
        ADX = dx[n:2*n].mean()
        
        ADX_series = pd.Series(np.nan, high_prices.index)

        # Calculate the following ADX with smoothing
        for idx, dx in enumerate(dx[2*n:]):
            ADX = 1/n*(ADX * (n-1) + dx)
            ADX_series[idx + 2*n] = ADX
        
        return ADX_series
    
    @staticmethod
    def calculate_CCI(
                    high_prices: pd.Series,
                    low_prices: pd.Series,
                    close_prices: pd.Series,
                    ndays: int = 20
                    ):
        """
        Calculate the Commodity Channel Index (CCI) for a given time series 
        containing high, low and close prices. (See: 
        'Commodity Channel Index: Tool for Trading Cyclic Trends' by D.R. 
        Lambert)
        
        Params:
            - high_prices (pd.Series): high values of the stock time series
            - low_prices (pd.Series): high values of the stock time series
            - close_prices (pd.Series): high values of the stock time series
            - ndays (int): The number of days to consider for the CCI calculation.
        
        Returns:
            - pd.Series: series containing the CCI index
        """
        # Calculate the Typical Price (TP)
        typical_price = (high_prices + low_prices + close_prices) / 3
        
        # Calculate the rolling mean of the Typical Price (TP)
        moving_average_tp = typical_price.rolling(ndays).mean()
        
        # Calculate the Mean Deviation (MD)
        mean_deviation = typical_price.rolling(ndays).apply(
                                            lambda x: np.mean(np.abs(x - np.mean(x)))
                                            )

        # Return CCI
        return (typical_price - moving_average_tp) / (0.015 * mean_deviation)
