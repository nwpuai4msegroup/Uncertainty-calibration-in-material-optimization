import sys
import pandas as pd
sys.path.append('..')
import ActiveLearning

df = pd.read_csv('data_S4.csv')
a = ActiveLearning.AL(df.iloc[:, 1:],
                      target_feature='Creep rupture life (h)',
                      n_test=0.2,
                      n_target=1,
                      n_init=0.1,
                      n_batch=1,
                      n_iter=50,
                      n_trials=1,
                      random_state=123,
                      regressors=['SVR'],
                      utility_functions=['EI'],
                      uq_metrics=['scc', 'mis_area', 'ence', 'cv'],
                      model_metrics=['mae', 'r2', 'mse', 'mape', 'rmse'],
                      al_metrics=['oc'],
                      recalibration=False,
                      save_results=True,
				file_name='Creep rupture life (h)_SVR_EI_1.csv'
                      )
a.run_trials()
