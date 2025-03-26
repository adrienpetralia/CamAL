import numpy as np
import pickle
import heapq
import time
import warnings
import torch

from src.helpers.class_activation_map import CAM
from src.helpers.data_processing import nilmdataset_to_clfdataset, undersampler
from src.helpers.torch_dataset import SimpleDataset, TSDataset
from src.helpers.torch_trainer import BasedClassifTrainer
from src.helpers.other import NILMmetrics, Classifmetrics

from src.models.camal.classifiers.camal_resnet import CamALResNet

from torch.utils.data import DataLoader


class CamAL(object):
    def __init__(
        self,
        n_predictors=5,
        device="cpu",
        batch_inference=128,
        loc_metrics=NILMmetrics(),
        clf_metrics=Classifmetrics(),
        **resnet_kwargs,
    ):
        self.device = device
        self.n_predictors = n_predictors

        self.batch_inference = batch_inference
        self.loc_metrics = loc_metrics
        self.clf_metrics = clf_metrics

        resnet_kwargs.setdefault("in_channels", 1)  # Default in_channels
        resnet_kwargs.setdefault("mid_channels", 64)  # Default mid_channel
        resnet_kwargs.setdefault("norm", "BatchNorm")  # Default mid_channel

        self.resnet_kwargs = resnet_kwargs

        self.is_fitted = False
        self.camal_logs = None
        self.list_resnet_predictors = None
        self.list_cam_predictors = None

    def train(
        self,
        train_dataset: tuple,
        valid_dataset: tuple,
        test_dataset: tuple,
        list_kernel_sizes: list = [5, 7, 9, 15, 25],
        n_try_by_kernel: int = 1,
        path: str = None,
        **training_kwargs,
    ):
        """
        Train CamAL ResNet ensemble.

        Args:

        Returns:
            None
        """

        assert n_try_by_kernel * len(list_kernel_sizes) >= self.n_predictors, (
            "The number of keep predictors does not meet the number of kernel to try and the try by kernel."
        )

        training_kwargs.setdefault("batch_size", 128)  # Default batch_size
        training_kwargs.setdefault("epochs", 50)  # Default number of epochs
        training_kwargs.setdefault("lr", 1e-3)  # Default lr
        training_kwargs.setdefault("weight_decay", 0)  # Default wd
        training_kwargs.setdefault("patience_es", 5)  # Default patience for es
        training_kwargs.setdefault("patience_rlr", 3)  # Default patience for reduce lr
        training_kwargs.setdefault("n_warmup_epochs", 1)  # Default warmup epochs
        training_kwargs.setdefault(
            "all_gpu", False
        )  # Default DataParallel call (True means using multiple GPUs for training)

        X_train, y_train = train_dataset[0], train_dataset[1]
        X_valid, y_valid = valid_dataset[0], valid_dataset[1]
        X_test, y_test = test_dataset[0], test_dataset[1]

        # Balance data class for training
        # if balance_class:
        X_train, y_train = undersampler(
            X_train, y_train, sampling_strategy="auto", seed=0
        )

        # Create dataset
        train_dataset = TSDataset(X_train, y_train)
        valid_dataset = TSDataset(X_valid, y_valid)
        test_dataset = TSDataset(X_test, y_test)

        # Init dicts result
        tmp_loss_results = {}
        camal_logs = {}
        camal_logs["ensemble_training_logs"] = {}

        start_training_time = time.time()

        idx_clf = 0
        for kernel_size in list_kernel_sizes:
            for nth_try in range(n_try_by_kernel):
                resnet_inst = CamALResNet(kernel_size=kernel_size, **self.resnet_kwargs)

                # Dataloader
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=training_kwargs["batch_size"],
                    shuffle=True,
                )
                valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=1, shuffle=False
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1, shuffle=False
                )

                # Init trainer
                clf_trainer = BasedClassifTrainer(
                    resnet_inst,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    learning_rate=training_kwargs["lr"],
                    weight_decay=training_kwargs["weight_decay"],
                    patience_es=training_kwargs["patience_es"],
                    patience_rlr=training_kwargs["patience_rlr"],
                    n_warmup_epochs=training_kwargs["n_warmup_epochs"],
                    device=self.device,
                    all_gpu=training_kwargs["all_gpu"],
                )

                # Train
                clf_trainer.train(training_kwargs["epochs"])

                # Eval
                clf_trainer.restore_best_weights()
                loss_eval, _ = clf_trainer.evaluate(test_loader)

                print(
                    f"Trained predictor {idx_clf} with kernel size: {kernel_size} (nth_try {nth_try}) - loss: {loss_eval}."
                )

                # Save loss value
                tmp_loss_results[f"Predictor_{idx_clf}"] = loss_eval
                camal_logs["ensemble_training_logs"][f"Predictor_{idx_clf}"] = (
                    clf_trainer.log
                )
                camal_logs["ensemble_training_logs"][f"Predictor_{idx_clf}"][
                    "kernel_size"
                ] = kernel_size
                camal_logs["ensemble_training_logs"][f"Predictor_{idx_clf}"][
                    "nth_try"
                ] = nth_try

                idx_clf += 1

        camal_logs["training_time"] = round((time.time() - start_training_time), 3)

        # Rank Resnet predictos accroding to eval loss
        heap = [(loss, name) for name, loss in tmp_loss_results.items()]
        heapq.heapify(heap)
        smallest = heapq.nsmallest(self.n_predictors, heap)
        del tmp_loss_results

        # Get the name of the best clf and select them for final ensemble
        list_best_clf = [name for _, name in smallest]

        camal_logs["camal_predictors_id"] = list_best_clf

        for i in range(idx_clf):
            if f"Predictor_{i}" not in list_best_clf:
                camal_logs["ensemble_training_logs"][f"Predictor_{i}"].pop(
                    "best_model_state_dict", None
                )

        camal_logs["is_fitted"] = True

        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(camal_logs, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.camal_logs = camal_logs
        self.is_fitted = True
        self.eval()

        return

    def test(self, data_test, appliance_mean_on_power=None, scaler=None):
        if len(data_test.shape) < 4:
            raise ValueError(
                "Provided test data need to be given in NILM standard (4D numpy array)"
            )

        # Create soft label
        soft_label, prob_detect = self.predict(data_test[:, 0, 0, :], return_prob=True)

        # Get y_state and y_hat_state
        y_state = data_test[:, 1, 1, :].ravel().astype(dtype=int)
        y_hat_state = soft_label.ravel().astype(dtype=int)

        if appliance_mean_on_power is not None:
            if scaler is not None:
                data_test_rescale = scaler.inverse_transform(data_test)
            warnings.warn(
                "No scaler provided for inverse_transform, are you sure you didn't scale the data for training?"
            )
            # Get true appliance power

            tmp_agg = data_test_rescale[:, 0, 0, :].ravel()
            y = data_test_rescale[:, 1, 0, :].ravel()

            # Create soft label power value according to mean appliance power param
            y_hat = y_hat_state * appliance_mean_on_power
            # Ensure that app consumption doesn't exceed aggregate
            y_hat[y_hat > tmp_agg] = tmp_agg[y_hat > tmp_agg]
            del data_test_rescale
            del tmp_agg

            # Compute metric (NILM reg + classif)
            metric_softlabel = self.loc_metrics(y, y_hat, y_state, y_hat_state)
        else:
            # Compute metric (NILM classif)
            metric_softlabel = self.loc_metrics(
                y=None, y_hat=None, y_state=y_state, y_hat_state=y_hat_state
            )

        _, y_clf_true = nilmdataset_to_clfdataset(data_test)
        metric_classif = self.clf_metrics(
            y=y_clf_true.ravel(), y_hat=prob_detect.ravel()
        )

        return {
            "Localization metrics:",
            metric_softlabel,
            "Classification metrics:",
            metric_classif,
        }

    def load(self, path: str):
        with open(path, "rb") as handle:
            self.camal_logs = pickle.load(handle)

        print("Log successfully loaded.")

        self.is_fitted = self.camal_logs["is_fitted"]

        return

    def predict(
        self,
        data: np.array,
        y: np.array = None,
        return_prob: bool = False,
        w_ma: int = 5,
    ):
        """
        Predict the localization activation of the appliance.

        Args:
            data (numpy.array): Input aggregate consumption data. This can be a 1D or 2D (batched) array.
            y (numpy.array, optional): 1D array, ground truth classifictaion label associated to the provided data
            return_prob (bool): Device type (i.e., cpu, cuda, cuda:0, etc.)

        Returns:
            None
        """

        if not self.is_fitted:
            raise ValueError(
                "CamAL instance is not fitted. Train CamAL or load a fitted instance of CamAL before trying to predict."
            )

        assert len(data.shape) <= 2, (
            f"Input data need to be a 1D or 2D (batched) numpy array, got a {len(data.shape)}D array of size {data.shape}."
        )

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)

        if self.list_resnet_predictors is None:
            self.eval()

        prediction = np.zeros_like(data)
        prob_detect = np.zeros((len(data), 1))

        # Loop on BestResNets
        for ind_res, resnet_inst in enumerate(self.list_resnet_predictors):
            # If running on GPUs: get first the per subsequence class labels to speed up inference
            if (self.device != "cpu") and (y is None):
                y = self._predict_subsequence_class(resnet_inst, data)

            for idx in range(len(data)):
                # Check subsequence's predicted label
                if y is not None:
                    if y[idx] < 1:
                        # Continue if appliance not detected
                        continue

                cam, y_pred, proba = self.list_cam_predictors[ind_res].run(
                    instance=data[idx], returned_cam_for_label=1
                )
                prob_detect[idx][0] += proba[1]

                # Or if app detected in this window
                if y_pred > 0:
                    # Clip CAM and MaxNormalization (between 0 and 1)
                    clip_cam = np.clip(cam, a_min=0, a_max=None)
                    clip_cam = np.nan_to_num(
                        clip_cam.astype(np.float32), nan=0.0, neginf=0.0, posinf=0.0
                    )

                    if clip_cam.max() > 0:
                        clip_cam = clip_cam / clip_cam.max()
                        prediction[idx] += clip_cam.ravel()

        # Majority voting: if the ensemble probability of detection < 0.5 appliance not detected in wins, soft label set to 0
        prob_detect = prob_detect / len(self.list_resnet_predictors)
        prediction = prediction / len(self.list_resnet_predictors)

        # Set window soft label to 0 if the ensemble not detect an appliance in a window
        prediction = prediction * np.round(prob_detect)

        # Sigmoid-Attention Module
        prediction = self.attention_sigmoid_module(data, prediction, w_ma=w_ma)

        if return_prob:
            return prediction, prob_detect
        else:
            return prediction

    def __call__(self, data: np.array, y: np.array = None, return_prob: bool = False):
        """
        Wrapper to predict function

        Args:
            device (str): Device type (i.e., cpu, cuda, cuda:0, etc.)

        Returns:
            None
        """
        return self.predict(data, y=y, return_prob=return_prob)

    def to(self, device: str) -> None:
        """
        Move ResNet predictors to a choosen device.

        Args:
            device (str): Device type (i.e., cpu, cuda, cuda:0, etc.)

        Returns:
            None
        """
        if self.list_resnet_predictors is not None:
            for predictors in self.list_resnet_predictors:
                predictors.to(device)
        else:
            warnings.warn("Cannot move anything to a device: no predictors in cache.")

        self.device = device

        return

    def eval(
        self,
    ) -> None:
        """Set ResNets ensemble predictors in eval mode (pytorch)

        Args:
            None
        Returns:
            None
        """

        if not self.is_fitted:
            raise ValueError(
                "CamAL instance is not fitted. Train CamAL or load a fitted instance of CamAL before trying to move it in eval mode."
            )

        list_resnet_predictors = []
        list_cam_predictors = []

        for resnet_name in self.camal_logs["camal_predictors_id"]:
            resnet_inst = CamALResNet(
                kernel_size=self.camal_logs["ensemble_training_logs"][resnet_name][
                    "kernel_size"
                ]
            )
            resnet_inst.to(self.device)

            resnet_inst.load_state_dict(
                self.camal_logs["ensemble_training_logs"][resnet_name][
                    "best_model_state_dict"
                ]
            )
            resnet_inst.eval()

            cam_resnet_inst = CAM(
                model=resnet_inst,
                device=self.device,
                last_conv_layer=resnet_inst._modules["layers"][2],
                fc_layer_name=resnet_inst._modules["linear"],
            )

            list_resnet_predictors.append(resnet_inst)
            list_cam_predictors.append(cam_resnet_inst)

        self.list_resnet_predictors = list_resnet_predictors
        self.list_cam_predictors = list_cam_predictors

        return

    def attention_sigmoid_module(
        self, data: np.array, soft_label: np.array, w_ma: int = 5
    ) -> np.array:
        """CamAL's attention sigmoid module.

        Args:
            data (numpy.array): The input data array containing aggregate power subsequences.
            soft_label (numpy.array): The array containing the computed average CAM.
            w_ma (int, optional): Moving average window length, defaults to "5".

        Returns:
            numpy.array
        """
        # Apply moving average
        soft_label = np.apply_along_axis(
            lambda x: self.moving_average(x, w=w_ma), axis=1, arr=soft_label
        )

        # Apply sigmoid on the product of averaged soft labels
        soft_label = self.sigmoid(soft_label * data)

        # Thresholding to obtain binary labels
        return np.round(soft_label)

    def moving_average(self, x: np.array, w: int) -> np.array:
        """Wrapper for moving average function based on the convolve numpy function

        Args:
            x (numpy.array): The input array.
            w (int): Moving average window length.

        Returns:
            numpy.array
        """
        return np.convolve(x, np.ones(w), "same") / w

    def sigmoid(self, z):
        """
        Rectified sigmoid function

        Args:
            z (numpy.array): array

        Returns:
            numpy.array
        """
        return 2 * (1.0 / (1.0 + np.exp(-z))) - 1

    def _predict_subsequence_class(self, model, data) -> np.array:
        # Private function that compute the classification label for each subsequence to speed up inference for large dataset
        model.to(self.device)
        model.eval()
        data = SimpleDataset(data)
        loader = DataLoader(data, batch_size=self.batch_inference, shuffle=False)

        y_hat = np.array([])
        with torch.no_grad():
            for ts in loader:
                logits = model(torch.Tensor(ts.float()).to(self.device))
                _, predicted = torch.max(logits, 1)

                y_hat = (
                    np.concatenate((y_hat, predicted.detach().cpu().numpy().flatten()))
                    if y_hat.size
                    else predicted.detach().cpu().numpy().flatten()
                )

        return y_hat.astype(np.int8)
