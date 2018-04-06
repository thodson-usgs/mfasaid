import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable

from linearmodel import stats, model as saidmodel


class HierarchicalModel:
    """Class combines mulitple surrogate models and uses the best among
    them to generate predicitons.

    Examples
    --------
    Given DataManagers containg constituent (con_data) and surrogates (sur_data),
    and columns: 'SSC', 'Turb', 'Discharge', initialize a HierarchicalModel as follows.

    >>> model_list = [
        ['SSC', ['log(Turb)', 'log(Discharge']],
        ['SSC', ['log(Turb)']]

    >>> model = HierarchicalModl(con_data, sur_data,
                                 model_list=model_list)

    >>> model.summary()

    TODO
    ----
    -model skill is currently assessed by r^2. Include alternative metrics
    like tightest prediction interval.
    -improve interface with linearmodel to a avoid reliance on
    private methods.

    """

    def __init__(self, constituent_data, surrogate_data, model_list,
                 min_samples=10, max_extrapolation=0.1, match_time=30,
                 p_thres=0.05):
        """ Initialize a HierarchicalModel

        :param constituent_data:
        :type constituent_data: DataManager
        :param surrogate_data:
        :type surrogate_data: DataManger
        :param model_list:
        """
        self._constituent_data = copy.deepcopy(constituent_data)
        self._surrogate_data = copy.deepcopy(surrogate_data)
        self._model_list = model_list

        self.match_time=match_time
        self.max_extrapolation= max_extrapolation
        self.min_samples = min_samples
        self.p_thres = p_thres



        self._create_models()


    def _create_models(self):
        """Populate a HierarchicalModel with SurrogateRatingModel objects.

        Called by __init__.

        """

        self._set_variables_and_transforms()
        #specify (n) the number of models managed within the instance
        n = len(self._model_list)
        self._models = [None for i in range(n)]

        #initialize arrays to store p values, nobs and rsquared of each model
        self._pvalues = np.zeros(n)
        self._nobs = np.zeros(n)
        self._rsquared = np.zeros(n)

        for i in range(n):
            #FIXME try to fix this by taking a the set of surrogate_variables
            surrogate_set = list(set(self._surrogates[i])) #removes duplicates
            self._models[i] = SurrogateRatingModel(self._constituent_data,
                                                   self._surrogate_data,
                                                   constituent_variable = self._constituent,
                                                   surrogate_variables = surrogate_set,
                                                   match_method = 'nearest',
                                                   #should set match in init
                                                   match_time = self.match_time)

            for surrogate in surrogate_set:
                #ceate an index of each occurance of the surrogate
                surrogate_transforms = [self._surrogate_transforms[i][j] for j,v in enumerate(self._surrogates[i]) if v == surrogate]
                #set the surrogate transforms based on the surrogate index
                self._models[i].set_surrogate_transform(surrogate_transforms, surrogate_variable=surrogate)

            #set transforms for each surrogate
            #for surrogate in self._surrogates[i]:
            #    self._models[i].set_surrogate_transform(self._surrogate_transforms[i], surrogate_variable=surrogate)

            self._models[i].set_constituent_transform(self._constituent_transforms[i])

            #FIXME depends on private methods
            res = self._models[i]._model._model.fit()
            self._pvalues[i] = res.f_pvalue
            self._rsquared[i] = res.rsquared_adj
            self._nobs[i] = res.nobs
            #TODO check transforms


    def _plot_model_time_series(self, ax, color='blue'):
        """Plots the time series predicted by the HierarchicalModel

        :param ax:
        :return:
        """
        #get all predicted data
        prediction = self.get_prediction()

        #plot mean response
        ax.plot(prediction.index,
                prediction[self._constituent].as_matrix(),
                color=color, linestyle='None',
                marker='.', markersize=5,
                label='Predicted ' + self._constituent)

        #lower prediction interval
        lower_interval_name = self._constituent + '_L90.0'

        #upper prediction interval
        upper_interval_name = self._constituent + '_U90.0'

        ax.fill_between(prediction.index,
                        prediction[lower_interval_name].as_matrix(),
                        prediction[upper_interval_name].as_matrix(),
                        facecolor='gray', edgecolor='gray', alpha=0.5, label='90% Prediction interval')


    def _set_variables_and_transforms(self):
        """Parses surrogates, constituent, and their transforms.

        This function is a part of HierarchicalModel's init.
        """
        self._constituent = None
        self._constituent_transforms = []

        self._surrogates = []
        self._surrogate_transforms = []

        temp_constituents = []

        for constituent, surrogates in self._model_list:

            constituent_transform, raw_constituent = saidmodel.find_raw_variable(constituent)
            temp_constituents.append(raw_constituent)
            self._constituent_transforms.append(constituent_transform)

            # make temporary list to store surrogates before appending them
            sur_temp = []
            sur_trans_temp = []
            for surrogate in surrogates:
                surrogate_transform, raw_surrogate = saidmodel.find_raw_variable(surrogate)
                sur_temp.append(raw_surrogate)
                sur_trans_temp.append(surrogate_transform)

            self._surrogates.append(sur_temp)
            self._surrogate_transforms.append(sur_trans_temp)

        # check that there is one and only one constituent

        temp_constituents =  list(set(temp_constituents))

        if len(temp_constituents) != 1:
            raise Exception('Only one raw constituent allowed')

        self._constituent = temp_constituents[0]


    def get_prediction(self, explanatory_data=None):
        """Use the HierarchicalModel to make a prediction based on explanatory_data.

        If no explanatory data is given, the prediction is based on the data uses to initialize
        the HierarchicalModel

        :param explanatory_data:
        :return:
        """
        #rank models by r2, starting with the lowest (worst)
        model_ranks = self._rsquared.argsort()

        for i in model_ranks:
            #skip models that aren't robust
            #TODO replace hard nobs thresh with thres * (surrogatecount + 1)
            if self._nobs[i] < 10 or self._pvalues[i] > self.p_thres:
                continue

            elif type(explanatory_data) is type(None):
                explanatory_data = self._models[i]._surrogate_data

            prediction = self._models[i]._model.predict_response_variable(
                explanatory_data = explanatory_data,
                raw_response=True,
                bias_correction=True,
                prediction_interval=True)

            try:
                hierarchical_prediction = hierarchical_prediction.update(prediction)

            except NameError:
                hierarchical_prediction = prediction

        return prediction


    def plot(self, plot_type='time series', ax=None, **kwargs):
        """Plots a HierarchicalModel

        :param plot_type:
        :param ax:
        :return:
        """

        # create an axis if one isn't passed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if plot_type == 'time series':
            self._plot_model_time_series(ax)
            #check if 

        else:
            #TODO: implement other plotting functions for HierarchicalModel
           raise ValueError("plot type not recognized or implemented")

        return ax


    def get_model_summaries(self):
        """Creates a table of summary stats describing each submodel
        """

        for model in self._models:
            model.get_model_summary()


    def get_model_reports(self):
        """Generates a model report for each individual model.

        WARNING: This may generate a lot of text.
        """
        for model in self._models:
            model.get_model_dataset()

    def summary(self):
        """Generates a summary table with basic statistics for each submodel.

        TODO: immprove interface with linearmodel, so that this doesn't rely on
        private methods.
        """
        summary = Summary()
        headers = ['Model form', 'Observations', 'Adjusted r^2',
                   'P value']
        table_data = []
        # for each model
        for model in self._models:
            row = []
            # populate row with model statistics
            res = model._model._model.fit()
            row.append(model._model.get_model_formula())
            row.append( round(res.nobs) )
            row.append( round(res.rsquared_adj, 2))
            row.append( format(res.f_pvalue, '.1E'))
            # append the row to the data
            table_data.append(row)

        # create table with data and headers:w
        table = SimpleTable(data=table_data, headers=headers)
        # add table to summary
        summary.tables.append(table)

        return summary


class SurrogateRatingModel:

    def __init__(self, constituent_data, surrogate_data, **kwargs):
        """

        :param constituent_data: 
        :type constituent_data: DataManager
        :param surrogate_data: 
        :type surrogate_data: DataManager
        """

        self._constituent_data = copy.deepcopy(constituent_data)
        self._surrogate_data = copy.deepcopy(surrogate_data)

        surrogate_variables = self._surrogate_data.get_variable_names()
        constituent_variables = self._constituent_data.get_variable_names()

        self._surrogate_variables = kwargs.pop('surrogate_variables', [surrogate_variables[0]])
        self._constituent_variable = kwargs.pop('constituent_variable', constituent_variables[0])

        self._surrogate_transform = dict(zip(surrogate_variables, [(None,)]*len(surrogate_variables)))
        self._constituent_transform = dict(zip(constituent_variables, [None]*len(constituent_variables)))

        match_method = kwargs.pop('match_method', 'nearest')
        self._match_method = dict(zip(surrogate_variables, [match_method]*len(surrogate_variables)))

        match_time = kwargs.pop('match_time', 0)
        self._match_time = dict(zip(surrogate_variables, [match_time]*len(surrogate_variables)))

        self._excluded_observations = pd.DatetimeIndex([])

        self._model = self._create_model()

    def _create_model(self):
        """
        
        :return: 
        """

        model_data = self._get_model_data()

        # get the name of the response variable to use in a model
        constituent_variable_transform = self.get_constituent_transform()
        response_variable = saidmodel.LinearModel.get_variable_transform_name(self._constituent_variable,
                                                                              constituent_variable_transform)

        # if more than one surrogate variable, create a multiple linear regression model
        if len(self._surrogate_variables) > 1:

            explanatory_variables = []

            # get the names of the
            for variable in self._surrogate_variables:
                variable_transform = self._surrogate_transform[variable][0]
                transformed_variable = saidmodel.LinearModel.get_variable_transform_name(variable, variable_transform)
                explanatory_variables.append(transformed_variable)
            model = saidmodel.MultipleOLSModel(model_data,
                                               response_variable=response_variable,
                                               explanatory_variables=explanatory_variables)

        # otherwise, create a complex simple or complex linear model, depending on the amount of surrogate variable
        # transformations
        else:

            surrogate_variable = self._surrogate_variables[0]
            surrogate_variable_transform = self._surrogate_transform[surrogate_variable]

            # if more than one transform, create a complex linear model
            if len(surrogate_variable_transform) > 1:
                model = saidmodel.ComplexOLSModel(model_data,
                                                  response_variable=response_variable,
                                                  explanatory_variable=surrogate_variable)
                for transform in surrogate_variable_transform:
                    model.add_explanatory_var_transform(transform)

            # otherwise, create a simple linear model
            else:
                explanatory_variable = \
                    saidmodel.LinearModel.get_variable_transform_name(surrogate_variable,
                                                                      surrogate_variable_transform[0])
                model = saidmodel.SimpleOLSModel(model_data,
                                                 response_variable=response_variable,
                                                 explanatory_variable=explanatory_variable)

        model.exclude_observation(self._excluded_observations)

        return model

    def _get_model_data(self):
        """
        
        :return: 
        """

        model_data = copy.deepcopy(self._constituent_data)

        for variable in self._surrogate_variables:
            if variable in model_data.get_variable_names():
                continue
            time_window_width = self._match_time[variable]
            match_method = self._match_method[variable]
            model_data = model_data.match_data(self._surrogate_data, variable, time_window_width, match_method)

        return model_data

    def _plot_constituent_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        model_dataset = self._model.get_model_dataset()

        constituent_variable_series = model_dataset[self._constituent_variable]

        missing_observation_index = model_dataset['Missing']
        excluded_observation_index = model_dataset['Excluded']
        included_observation_index = ~(missing_observation_index | excluded_observation_index)

        if excluded_observation_index.any():
            ax.plot(constituent_variable_series.index[excluded_observation_index],
                    constituent_variable_series[excluded_observation_index].as_matrix(),
                    marker='s', color='red', linestyle='None', label='Observations excluded from model')
        if missing_observation_index.any():
            ax.plot(constituent_variable_series.index[missing_observation_index],
                    constituent_variable_series[missing_observation_index].as_matrix(),
                    marker='d', color='black', linestyle='None', label='Observations missing from model')
        if included_observation_index.any():
            ax.plot(constituent_variable_series.index[included_observation_index],
                    constituent_variable_series[included_observation_index].as_matrix(),
                    marker='o', markerfacecolor='yellow', markeredgecolor='black',
                    linestyle='None', label='Observations included in model')

    def _plot_model_time_series(self, ax):
        """

        :return:
        """

        # plot the predicted time series
        self._plot_predicted_time_series(ax)
        self._plot_constituent_time_series(ax)

        # set the y scale to logarithmic if the response variable is log transformed
        constituent_transform = self.get_constituent_transform()
        if (constituent_transform is 'log') or (constituent_transform is 'log10'):
            ax.set_yscale('log')

        ax.set_ylabel(self._constituent_variable)
        ax.legend(loc='best', numpoints=1)

    def _plot_predicted_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        # get predicted data to plot
        predicted_data = self._model.predict_response_variable(
            explanatory_data=self._surrogate_data, raw_response=True, bias_correction=True, prediction_interval=True)

        # mean response
        ax.plot(predicted_data.index, predicted_data[self._constituent_variable].as_matrix(), color='blue',
                linestyle='None', marker='.', markersize=5,
                label='Predicted ' + self._constituent_variable)

        # lower prediction interval
        lower_interval_name = self._constituent_variable + '_L90.0'

        # upper prediction interval
        upper_interval_name = self._constituent_variable + '_U90.0'

        ax.fill_between(predicted_data.index,
                        predicted_data[lower_interval_name].as_matrix(),
                        predicted_data[upper_interval_name].as_matrix(),
                        facecolor='gray', edgecolor='gray', alpha=0.5, label='90% Prediction interval')

    def _plot_observation_quantile(self, surrogate_variable_name, ax):
        """

        :param surrogate_variable_name:
        :param ax:
        :return:
        """

        # get the surrogate variable plotting positions
        surrogate_variable = self._surrogate_data.get_variable(surrogate_variable_name)
        surrogate_variable = surrogate_variable.dropna()
        surrogate_variable = np.squeeze(surrogate_variable)
        surrogate_pp = stats.calc_plotting_position(surrogate_variable, a=0)

        # plot the surrogate plotting positions
        ax.plot(surrogate_variable, surrogate_pp, marker='.', markersize=5,
                linestyle='None', label='Surrogate observation')

        # find the information to plot model observations dates included in the model
        model_dataset = self._model.get_model_dataset()
        missing_or_excluded_index = model_dataset['Missing'] | model_dataset['Excluded']
        included_dates = model_dataset.index[~missing_or_excluded_index]

        # find the surrogate values (and plotting position) closest-in-time to model observations
        model_obs_surrogate_value = np.empty(included_dates.shape)
        model_obs_pp = np.empty(included_dates.shape)
        for i in range(len(included_dates)):

            # find the minimum absolute time difference
            abs_time_diff = np.abs(included_dates[i] - surrogate_variable.index)
            min_abs_time_diff = np.min(abs_time_diff)

            # get the closest observation index
            closest_obs_index = abs_time_diff == min_abs_time_diff

            # set the surrogate value and plotting position to the earliest occurrence
            model_obs_surrogate_value[i] = surrogate_variable.ix[closest_obs_index][0]
            model_obs_pp[i] = surrogate_pp[closest_obs_index][0]

        # plot model observation occurrences
        ax.plot(model_obs_surrogate_value, model_obs_pp,
                marker='o', markerfacecolor='yellow', markeredgecolor='black',
                linestyle='None', label='Model observation')

        ax.set_xlabel(surrogate_variable_name)
        ax.set_ylabel('Cumulative frequency')

        ax.legend(loc='best', numpoints=1)

    def add_surrogate_transform(self, surrogate_variable, surrogate_transform):
        """

        :param surrogate_variable:
        :param surrogate_transform: 
        :return: 
        """

        saidmodel.LinearModel.check_transform(surrogate_transform)

        surrogate_variables = self._surrogate_data.get_variable_names()
        if surrogate_variable not in surrogate_variables:
            raise ValueError("Invalid surrogate variable name: {}".format(surrogate_variable))

        self._surrogate_transform[surrogate_variable] = self._surrogate_transform[surrogate_variable] \
                                                        + (surrogate_transform,)

        if surrogate_variable in self._surrogate_variables:
            self._model = self._create_model()

    def exclude_observations(self, observations):
        """

        :param observations: 
        :return: 
        """

        self._excluded_observations = observations
        self._model = self._create_model()

    def get_constituent_transform(self):
        """ Returns the current transform of the constituent variable

        :return: 
        """

        return self._constituent_transform[self._constituent_variable]

    def get_constituent_variable(self):
        """

        :return: 
        """

        return self._constituent_variable

    def get_constituent_variable_names(self):
        """
        
        :return: 
        """

        return self._constituent_data.get_variable_names()

    def get_explanatory_variables(self):

        return self._model.get_explanatory_variables()

    def get_response_variable(self):

        return self._model.get_response_variable()

    def get_surrogate_transform(self):
        """Returns a dictionary containing the surrogate variables and the transforms

        :return: 
        """

        return {var: self._surrogate_transform[var] for var in self._surrogate_variables}

    def get_surrogate_variable_names(self):
        """
        
        :return: 
        """

        return self._surrogate_data.get_variable_names()

    def get_surrogate_variables(self):
        """

        :return: 
        """

        return copy.deepcopy(self._surrogate_variables)

    def get_model_dataset(self):
        """

        :return:
        """
        return self._model.get_model_dataset()

    def get_model_report(self):
        """

        :return: 
        """

        return self._model.get_model_report()

    def plot(self, plot_type='model_scatter', ax=None, **kwargs):
        """

        :param plot_type:
        :param ax:
        :return:
        """

        # create an axis if one isn't passed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if plot_type == 'time series':
            self._plot_model_time_series(ax)
        elif plot_type == 'quantile':
            # TODO: figure out how combinations of kwargs can be enforced
            surrogate_variable = kwargs.pop('surrogate_variable', self._surrogate_variables[0])
            self._plot_observation_quantile(surrogate_variable, ax)
        else:
            self._model.plot(plot_type, ax)

        return ax

    def set_constituent_transform(self, constituent_transform):
        """

        :param constituent_transform: 
        :return: 
        """

        saidmodel.LinearModel.check_transform(constituent_transform)
        self._constituent_transform[self._constituent_variable] = constituent_transform
        self._model = self._create_model()

    def set_constituent_variable(self, constituent_variable):
        """
        
        :param constituent_variable: 
        :return: 
        """

        constituent_variable_names = self.get_constituent_variable_names()
        if constituent_variable not in constituent_variable_names:
            raise ValueError("Invalid constituent variable name: {}".format(constituent_variable))

        self._constituent_variable = constituent_variable

        self._model = self._create_model()

    def set_observation_match_method(self, method, time):
        """

        :param method: 
        :param time: 
        :return: 
        """

        if method not in ['nearest', 'mean']:
            raise ValueError("Invalid match method: {}".format(method))

        if time < 0:
            raise ValueError("time must be greater than or equal to zero.")

        match_method = dict(zip(self._surrogate_variables, [method]*len(self._surrogate_variables)))
        match_time = dict(zip(self._surrogate_variables, [time]*len(self._surrogate_variables)))

        self._match_method.update(match_method)
        self._match_time.update(match_time)

        self._model = self._create_model()

    def set_surrogate_transform(self, surrogate_transforms, surrogate_variable=None):
        """Set the surrogate transforms.

        :param surrogate_transforms: array-like
        :param surrogate_variable: string, optional
        :return:
        """

        for transform in surrogate_transforms:
            saidmodel.LinearModel.check_transform(transform)
        if surrogate_variable is None:
            surrogate_variable = self.get_surrogate_variables()[0]
        elif surrogate_variable not in self._surrogate_data.get_variable_names():
            raise ValueError("Invalid variable name: {}".format(surrogate_variable))

        self._surrogate_transform[surrogate_variable] = surrogate_transforms
        self._model = self._create_model()

    def set_surrogate_variables(self, surrogate_variables):
        """
        
        :param surrogate_variables: 
        :return: 
        """

        for variable in surrogate_variables:
            if variable not in self._surrogate_data.get_variable_names():
                raise ValueError("Invalid variable name: {}".format(variable))

        self._surrogate_variables = copy.deepcopy(list(surrogate_variables))
        self._model = self._create_model()
