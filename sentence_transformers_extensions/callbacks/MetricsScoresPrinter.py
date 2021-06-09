import pandas as pd
import seaborn as sns
from IPython.display import display, clear_output
from tqdm.auto import tqdm

def highlight_max(s):
    '''
    highlight the maximum in a Series as bold.
    '''
    is_max = s == s.max()
    return ['font-weight: bolder' if v else '' for v in is_max]

class MetricsScoresPrinter:
  def __init__(self, epochs, main_score_function='cos_sim'):
    self.epochs = epochs
    self.main_score_function = main_score_function
    self.scores_df = None
    self.bar = None
    self.start_datetime = pd.Timestamp.now()
  
  def __call__(self, metric_scores, epochs, steps):
    res_flat = {(split, metric, k): [v] for split, split_dict in metric_scores.scores.items() for metric, metric_dict in split_dict[self.main_score_function].items() for k, v in metric_dict.items()}
    score_df = pd.DataFrame(res_flat).assign(step=steps, epoch=epochs).set_index(['epoch', 'step'])
    current_datetime = pd.Timestamp.now()
    score_df[('statistics', 'time', 'timestamp')] = current_datetime
    score_df[('statistics', 'time', 'deltatime')] = pd.Timedelta(current_datetime - self.start_datetime)
    if self.scores_df is None:
      self.scores_df = score_df
    else:
      self.scores_df = self.scores_df.append(score_df)
    clear_output(wait=True)
    self.bar = tqdm(desc='EPOCHS', total=self.epochs, initial=epochs+1)
    display(self.scores_df.style
            .set_properties(**{'border-style':'solid', 'border-right': '0px', 'border-left': '0px', 'border-bottom': '0px', 'border-color': 'black', 'padding': '5px', 'border-collapse': 'collapse'})
            .background_gradient(cmap=sns.color_palette("RdYlGn", as_cmap=True), subset=list(self.scores_df.columns.values), vmin=0, vmax=1)
            .format(dict([(col, "{:.3}") for col in self.scores_df.columns.values if col[0] not in {'statistics'}] + [(('statistics', 'time', 'timestamp'), '{:%Y-%m-%d %H:%M:%S}')]))
            .apply(highlight_max))