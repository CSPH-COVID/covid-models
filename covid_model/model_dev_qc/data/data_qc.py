import pandas as pd
import numpy as np
from typing import Optional, List


class FunctionNotCalled(Exception):
    pass


class GenericQC(object):

    def __init__(self,name):
        self.name: str = name
        self._report_rows: List[str] = []
        self._result: bool = False
        self._has_run = False

    def report_df(self) -> Optional[pd.DataFrame]:
        # Should only be called once the function has been run, and a result is available
        if not self._has_run:
            raise FunctionNotCalled("Report DF cannot be generated until check() has been called.")

        # Generate a report dataframe on the fly

        tmp_df = pd.DataFrame({"Test": self.name,
                               "Pass": self._result,
                               "Issue #": np.arange(len(self._report_rows))+1,
                               "Explanation": self._report_rows})
        tmp_df.set_index(["Test", "Pass", "Issue #"], inplace=True)

        return tmp_df

    def check(self, *args, **kwargs):
        raise NotImplementedError


class NegativeValuesQC(GenericQC):

    def __init__(self):
        super(NegativeValuesQC, self).__init__(name="NegativeValuesQC")

    def check(self, data: pd.DataFrame):
        self._report_rows.clear()
        neg_values_exist = np.any(data < 0)
        if neg_values_exist:
            # Test fails if negative values exist
            self._result = False
            # Get idx of negative values
            neg_ind, neg_col = np.where(data < 0)
            # Index labels from DataFrame
            neg_ind_lbl = data.index[neg_ind]
            # Column labels from DataFrame
            neg_col_lbl = data.columns[neg_col]
            # Values from DataFrame
            neg_ind_val = data.values[neg_ind, neg_col]
            self._report_rows.extend(
                [f"Row: '{r}' Col: '{c}' Val: {v}" for r, c, v in zip(neg_ind_lbl, neg_col_lbl, neg_ind_val)])
        else:
            self._result = True
            self._report_rows.append("No negative values encountered.")
        self._has_run = True
        return self._result


class MissingValuesQC(GenericQC):

    def __init__(self):
        super(MissingValuesQC, self).__init__(name="MissingValuesQC")

    def check(self, data: pd.DataFrame):
        self._report_rows.clear()
        data_na = data.isna()
        missing_values = np.any(data_na)
        if missing_values:
            # Test fails if negative values exist
            self._result = False
            # Get idx of negative values
            na_ind, na_col = np.where(data_na)
            # Index labels from DataFrame
            na_ind_lbl = data.index[na_ind]
            # Column labels from DataFrame
            na_col_lbl = data.columns[na_col]
            self._report_rows.extend(
                [f"NA value in Row: '{r}' Col: '{c}'" for r, c in zip(na_ind_lbl, na_col_lbl)])
        else:
            self._result = True
            self._report_rows.append("No missing values encountered.")
        self._has_run = True
        return self._result


