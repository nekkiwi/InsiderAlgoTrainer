from src.training.feature_selector import FeatureSelector
from src.training.train import ModelTrainer
from src.backtesting.backtest import Backtester
    
def main():   
    ###################
    # Initializations #
    ###################
    
    feature_selector    = FeatureSelector()
    backtester          = Backtester()
    
    features_df_filtered    = None
    targets_df              = None
    
    ############
    # Training #
    ############

    # selected_features = feature_selector.run(features_df_filtered, targets_df, p_threshold=0.05)
    
    # models  = ['RandomForestOverSample']
    # targets = ['pos_return_1w_raw', 'pos_alpha_1w_raw', 'pos_return_1w_limstop', 'pos_alpha_1w_limstop', 
    #            'pos_return_1m_raw', 'pos_alpha_1m_raw', 'pos_return_1m_limstop', 'pos_alpha_1m_limstop']

    # for model in models: 
    #     for target in targets:
    #         ModelTrainer().run(target, model, selected_features, features_df_filtered, targets_df)
        
    ###############
    # Backtesting #
    ###############
    
    limit_array = [ 0.03]
    stop_array  = [-0.02]

    backtester.run(limit_array, stop_array)
      

if __name__ == "__main__":
    main()
