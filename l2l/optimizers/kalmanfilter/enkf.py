import numpy as np
import torch


class EnsembleKalmanFilter:
    def __init__(self, maxit=1):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013),
        The Ensemble Kalman Filter for Inverse Problems.
        doi:10.1088/0266-5611/29/4/045001

        :param maxit: int, maximum number of iterations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.Cpp = None
        self.Cup = None
        self.ensemble = None
        self.observations = None

        self.maxit = maxit
        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0

    def fit(self, ensemble, ensemble_size, observations, model_output, gamma):
        """
        Prediction and update step of the EnKF
        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k). E.g. network output activity
        :param  gamma: nd numpy array, Normalizes the model-data distance in the
            update step, :`noise * I` (I is identity matrix) or
            :math:`\\gamma=I`
        :return self, Possible outputs are:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the
                ensembles
        """
        # get shapes
        self.gamma_s, self.dims = _get_shapes(observations, model_output)

        if isinstance(gamma, (int, float)):
            if float(gamma) == 0.:
                self.gamma = np.eye(self.gamma_s)
        else:
            self.gamma = gamma

        # copy the data so we do not overwrite the original arguments
        self.ensemble = ensemble
        self.observations = observations
        # convert to pytorch
        self.ensemble = torch.as_tensor(
            self.ensemble, device=self.device, dtype=torch.float32)
        self.observations = torch.as_tensor(
            self.observations, device=self.device, dtype=torch.float32)
        self.gamma = torch.as_tensor(
            self.gamma, device=self.device, dtype=torch.float32)
        model_output = torch.as_tensor(
            model_output, device=self.device, dtype=torch.float32)

        for i in range(self.maxit):
            for d in range(model_output.shape[-1]):
                # Calculate the covariances
                mo = model_output[:, :, d]
                Cpp = _cov_mat(mo, mo, ensemble_size)
                Cup = _cov_mat(self.ensemble, mo, ensemble_size)
                self.ensemble = _update_step(self.ensemble,
                                             self.observations[d],
                                             mo, self.gamma, Cpp, Cup)
        return self


def _update_step(ensemble, observations, g, gamma, Cpp, Cup):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    cpg_inv = torch.inverse(Cpp + gamma)
    return torch.mm(torch.mm(Cup, cpg_inv), (observations - g).t()).t() + ensemble


def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    if x.shape[0] == 1:
        return torch.tensordot((x - x.mean()), (y - y.mean()),
                               dims=([0], [0])) / ensemble_size
    else:
        return torch.tensordot((x - x.mean(0)), (y - y.mean(0)),
                               dims=([0], [0])) / ensemble_size


def _get_shapes(observations, model_output):
    """
    Returns individual shapes

    :returns gamma_shape, length of the observation (here: size of last layer
                          of network)
    :returns dimensions, number of observations (and data)
    """
    if model_output.size > 2:
        gamma_shape = model_output.shape[1]
    else:
        gamma_shape = model_output.shape[0]
    dimensions = observations.shape[0]
    return gamma_shape, dimensions


def _one_hot_vector(index, shape):
    """
    Encode targets into one-hot representation
    """
    target = np.zeros(shape)
    target[index] = 1.0
    return target


def _encode_targets(targets, shape):
    return np.array(
        [_one_hot_vector(targets[i], shape) for i in range(targets.shape[0])])


def _shuffle(data, targets):
    """
    Shuffles the data and targets by permuting them
    """
    indices = np.random.permutation(targets.shape[0])
    return data[indices], targets[indices]
