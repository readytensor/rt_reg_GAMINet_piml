import os
import warnings
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from piml.models import GAMINetRegressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the GAMINetRegressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "GAMINetRegressor"

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        interact_num: int = 10,
        subnet_size_main_effect: tuple = (20,),
        subnet_size_interaction: tuple = (20, 20),
        activation_func: str = "ReLU",
        max_epochs: tuple[int, int, int] = (1000, 1000, 1000),
        learning_rates: tuple[int, int, int] = (0.001, 0.001, 0.0001),
        early_stop_thres: tuple[str, str, str] = ("auto", "auto", "auto"),
        batch_size: int = 1000,
        batch_size_inference: int = 10000,
        max_iter_per_epoch: int = 100,
        val_ratio: float = 0.2,
        warm_start: bool = True,
        gam_sample_size: int = 5000,
        mlp_sample_size: int = 1000,
        heredity: bool = True,
        reg_clarity: float = 0.1,
        loss_threshold: float = 0.01,
        reg_mono: float = 0.1,
        mono_increasing_list: tuple[str] = (),
        mono_decreasing_list: tuple[str] = (),
        mono_sample_size: int = 1000,
        include_interaction_list: tuple[str, str] = (),
        boundary_clip: bool = True,
        normalize: bool = True,
        verbose: bool = False,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        **kwargs,
    ):
        """
        Initializes the model with specified configurations.

        Parameters:
            feature_names (Optional[List[str]]): The list of feature names. Default is None.
            feature_types (Optional[List[str]]): The list of feature types. Available types include “numerical” and “categorical”. Default is None.
            interact_num (int): The max number of interactions to be included in the second stage training. Default is 10.
            subnet_size_main_effect (tuple): The hidden layer architecture of each subnetwork in the main effect block. Default is (20,).
            subnet_size_interaction (tuple): The hidden layer architecture of each subnetwork in the interaction block. Default is (20, 20).
            activation_func (str): The name of the activation function. Options are “ReLU”, “Sigmoid”, “Tanh”. Default is “ReLU”.
            max_epochs (tuple[int, int, int]): The max number of epochs in the first, second, and third stages, respectively. Default is (1000, 1000, 1000).
            learning_rates (tuple[float, float, float]): The initial learning rates for the Adam optimizer in the first, second, and third stages, respectively. Default is (0.001, 0.001, 0.0001).
            early_stop_thres (tuple[str, str, str]): The early stopping threshold in the first, second, and third stages, respectively. "auto" or an integer. Default is ("auto", "auto", "auto").
            batch_size (int): The batch size for training. Should not be larger than the training size * (1 - validation ratio). Default is 1000.
            batch_size_inference (int): The batch size used in the inference stage to avoid out-of-memory issues. Default is 10000.
            max_iter_per_epoch (int): The max number of iterations per epoch, making training scalable for very large datasets. Default is 100.
            val_ratio (float): The validation ratio, should be greater than 0 and smaller than 1. Default is 0.2.
            warm_start (bool): Initialize the network by fitting a rough B-spline based GAM model. Default is True.
            gam_sample_size (int): The sub-sample size for GAM fitting with warm_start=True. Default is 5000.
            mlp_sample_size (int): The generated sample size for individual subnetwork fitting with warm_start=True. Default is 1000.
            heredity (bool): Whether to perform interaction screening subject to heredity constraint. Default is True.
            reg_clarity (float): The regularization strength of marginal clarity constraint. Default is 0.1.
            loss_threshold (float): The loss tolerance threshold for selecting fewer main effects or interactions. Default is 0.01.
            reg_mono (float): The regularization strength of monotonicity constraint. Default is 0.1.
            mono_increasing_list (tuple[str]): The feature names subject to monotonic increasing constraint. Default is ().
            mono_decreasing_list (tuple[str]): The feature names subject to monotonic decreasing constraint. Default is ().
            mono_sample_size (int): The sample size for imposing monotonicity regularization. Default is 1000.
            include_interaction_list (tuple[str, str]): The tuple of interactions to be included for fitting. Default is ().
            boundary_clip (bool): Whether to clip the feature values by their min and max values in the training data during inference. Default is True.
            normalize (bool): Whether to normalize the data before inputting to the network. Default is True.
            verbose (bool): Whether to output the training logs. Default is False.
            n_jobs (int): The number of CPU cores for parallel computing. -1 means all available CPUs. Default is 1.
            device (str): The hardware device name used for training. Default is "cpu".
            random_state (int): The random seed. Default is 0.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            None.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.interact_num = interact_num
        self.subnet_size_main_effect = tuple(subnet_size_main_effect)
        self.subnet_size_interaction = tuple(subnet_size_interaction)
        self.activation_func = activation_func
        self.max_epochs = tuple(max_epochs)
        self.learning_rates = tuple(learning_rates)
        self.early_stop_thres = tuple(early_stop_thres)
        self.batch_size = batch_size
        self.batch_size_inference = batch_size_inference
        self.max_iter_per_epoch = max_iter_per_epoch
        self.val_ratio = val_ratio
        self.warm_start = warm_start
        self.gam_sample_size = gam_sample_size
        self.mlp_sample_size = mlp_sample_size
        self.heredity = heredity
        self.reg_clarity = reg_clarity
        self.loss_threshold = loss_threshold
        self.reg_mono = reg_mono
        self.mono_increasing_list = tuple(mono_increasing_list)
        self.mono_decreasing_list = tuple(mono_decreasing_list)
        self.mono_sample_size = mono_sample_size
        self.include_interaction_list = tuple(include_interaction_list)
        self.boundary_clip = boundary_clip
        self.normalize = normalize
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> GAMINetRegressor:
        """Build a new binary classifier."""
        model = GAMINetRegressor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            interact_num=self.interact_num,
            subnet_size_main_effect=self.subnet_size_main_effect,
            subnet_size_interaction=self.subnet_size_interaction,
            activation_func=self.activation_func,
            max_epochs=self.max_epochs,
            learning_rates=self.learning_rates,
            early_stop_thres=self.early_stop_thres,
            batch_size=self.batch_size,
            batch_size_inference=self.batch_size_inference,
            max_iter_per_epoch=self.max_iter_per_epoch,
            val_ratio=self.val_ratio,
            warm_start=self.warm_start,
            gam_sample_size=self.gam_sample_size,
            mlp_sample_size=self.mlp_sample_size,
            heredity=self.heredity,
            reg_clarity=self.reg_clarity,
            loss_threshold=self.loss_threshold,
            reg_mono=self.reg_mono,
            mono_increasing_list=self.mono_increasing_list,
            mono_decreasing_list=self.mono_decreasing_list,
            mono_sample_size=self.mono_sample_size,
            include_interaction_list=self.include_interaction_list,
            boundary_clip=self.boundary_clip,
            normalize=self.normalize,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            device=self.device,
            random_state=self.random_state,
            **self.kwargs,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"eta: {self.eta}, "
            f"gamma: {self.gamma}, "
            f"max_depth: {self.max_depth}, "
            f"n_estimators: {self.n_estimators})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
