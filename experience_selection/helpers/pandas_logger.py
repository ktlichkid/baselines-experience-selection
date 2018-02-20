from os import makedirs
from os.path import expanduser, join, isdir, isfile
import numpy as np
import pandas as pd


class LogVariable(object):
    def __init__(self, values):
        self.last_value = 0
        self.count_cum = 0
        self.sum_cum = 0
        self.max_cum = -np.inf
        self.min_cum = np.inf
        self.length = values

        self.reset_period()
        self.values = {
            'mean': np.zeros(values),
            'most_recent': np.zeros(values),
            'max': np.zeros(values),
            'min': np.zeros(values),
            'cum': np.zeros(values)
        }
        self.index = 0

    def __getitem__(self, item):
        return self.values[item]

    def reset_period(self):
        self.sum_period = 0
        self.count_period = 0
        self.min_period = np.inf
        self.max_period = -np.inf

    def log(self, value):
        self.last_value = value
        self.sum_cum += value
        self.sum_period += value
        self.count_cum += 1
        self.count_period += 1
        if value > self.max_cum:
            self.max_cum = value
        if value < self.min_cum:
            self.min_cum = value
        if value > self.max_period:
            self.max_period = value
        if value < self.min_period:
            self.min_period = value

    def next_period(self):
        if self.count_period > 0:
            self.values['mean'][self.index] = self.sum_period / self.count_period
            self.values['max'][self.index] = self.max_period
            self.values['min'][self.index] = self.min_period
            self.values['most_recent'][self.index] = self.last_value
            self.values['cum'][self.index] = self.sum_cum
        else:
            self.values['mean'][self.index] = np.NaN
            self.values['max'][self.index] = np.NaN
            self.values['min'][self.index] = np.NaN
            self.values['most_recent'][self.index] = np.NaN
            self.values['cum'][self.index] = self.sum_cum

        self.index += 1
        if self.index >= self.length:
            self.index = self.length - 1

        self.reset_period()


class PandaLogger(object):
    def __init__(self, experiment_definition, logfile_name='log_trial_{:03d}.csv', trial_idx=None):
        self._trial_idx = trial_idx
        self.logfile_name = logfile_name
        self.experiment_definition = experiment_definition
        self.log_def = experiment_definition['pd_log']
        self.data_def = {name[12:]: self.log_def['pd_log_data'][name] for name in self.log_def['pd_log_data']}
        self.interval_def = self.log_def['pd_log_interval']
        self.last_update_idx = 0
        self.data = self.create_data_dict()
        self.file_name = self.full_logfile_name

    def __getitem__(self, item):
        return self.data[item]

    @property
    def variable_names(self):
        fields = {self.update_field_name}
        for cat in self.data_def.values():
            fields = fields.union(set(cat))
        return fields

    def __len__(self):
        return self.interval_def['pd_log_interval_values']

    @property
    def full_logfile_name(self):
        return self.symbolic_name.format(self.trial_idx)

    @property
    def trial_idx(self):
        if self._trial_idx:
            return self._trial_idx
        else:
            sym_name, idx = self.symbolic_name, 0
            while isfile(sym_name.format(idx)):
                idx += 1
            self._trial_idx = idx
            return idx

    @property
    def symbolic_name(self):
        base_path = expanduser(self.experiment_definition['log_dir']) + self.experiment_definition[
            'variation_name'] + '/'
        if not isdir(base_path):
            makedirs(base_path)
        return base_path + self.logfile_name

    @property
    def update_field_name(self):
        return self.interval_def['pd_log_interval_field']

    @property
    def next_update_value(self):
        totalname = self.update_field_name + 's'
        assert totalname in self.experiment_definition, \
            'Since "{:s}" was the log criterion, there should be a field "{:s}" in the experiment ' \
            '' \
            '' \
            'definition to indicate the total number of {:s}.'.format(self.update_field_name,
                                                                      totalname, totalname)
        total = self.experiment_definition[totalname]
        return np.math.floor((self.last_update_idx + 1) * (total / len(self)))

    @property
    def requested_data(self):
        data = {self.update_field_name: self[self.update_field_name]['most_recent']}
        for datatype in ('mean', 'max', 'min', 'most_recent', 'cum'):
            if datatype in self.data_def:
                for varname in self.data_def[datatype]:
                    data[datatype + '_' + varname] = self[varname][datatype]
        return data

    def log(self, partial=False, **log_data):
        for key in log_data:
            self[key].log(log_data[key])
        if not partial:
            while log_data[self.update_field_name] >= self.next_update_value:
                self.report_period()

    def report_period(self):
        self.last_update_idx += 1
        for var in self.data.values():
            var.next_period()
        self.save_to_disk()

    def create_data_dict(self):
        return {name: LogVariable(len(self)) for name in self.variable_names}

    def save_to_disk(self):
        data = self.requested_data
        df = pd.DataFrame(data=data)
        df.to_csv(self.file_name)


class PandaLogChecker(PandaLogger):

    def __init__(self):
        self._trial_idx = None
        self.experiment_definition = None
        self.logfile_name = None

    def get_trial_idx(self, experiment_definition, logfile_name='log_trial_{:03d}.csv'):
        self._trial_idx = None
        self.experiment_definition = experiment_definition
        self.logfile_name = logfile_name
        return self.trial_idx
