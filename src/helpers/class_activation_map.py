import numpy as np
import torch
import torch.nn.functional as F

from torch import topk
from torch.autograd import Variable

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def fmax(val):
    if val > 0 : return val
    else: return 0
def fmin(val):
    if val < 0 : return val
    else: return 0
    

# ========================================= Class Activation Map ========================================= #
class CAM():
    def __init__(self, model, device, last_conv_layer='layer3', fc_layer_name='fc1', verbose=False):
        
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer_name = fc_layer_name
        self.model = model
        self.verbose = verbose


    def run(self, instance, label_instance=None, returned_cam_for_label=None):
        assert label_instance is not None or returned_cam_for_label is not None
        cam, label_pred, proba =  self.__get_CAM_class(np.array(instance), returned_cam_for_label)

        return cam, label_pred, proba

    # ================Private methods===================== #

    def __getCAM(self,feature_conv, weight_fc, class_idx):
        _, nc, length = feature_conv.shape
        feature_conv_new = feature_conv
        cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nc,length)))
        cam = cam.reshape(length)
        
        return cam


    def __get_CAM_class(self, instance, returned_cam_for_label):
        original_length = len(instance)

        instance_to_try = Variable(torch.tensor(instance.reshape((1, 1, original_length))).float().to(self.device), requires_grad=True)

        final_layer = self.last_conv_layer
        activated_features = SaveFeatures(final_layer)
        prediction = self.model(instance_to_try)
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        activated_features.remove()
        weight_softmax_params = list(self.fc_layer_name.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        
        class_idx = topk(pred_probabilities, 1)[1].int()
        
        if self.verbose:
            print('Pred proba :', pred_probabilities.cpu().numpy())

        if returned_cam_for_label is None:
            overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx)
        else:
            overlay = self.__getCAM(activated_features.features, weight_softmax, torch.Tensor(np.array([returned_cam_for_label])).int().to(self.device))
        
        return overlay, class_idx.item(), pred_probabilities.cpu().numpy()


# ========================================= Gradient Class Activation Map ========================================= #

class FeatureHooks():
    def __init__(self, verbose=False):
        self.gradients = None
        self.activations = None
        self.verbose = verbose

    def save_activations(self, module, input, output):
        self.activations = output
        if self.verbose:
            print(f'Forward hook running... Activations size: {self.activations.size()}')

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        if self.verbose:
            print(f'Backward hook running... Gradients size: {self.gradients.size()}')


class GradCAM():
    def __init__(self, model, device, last_conv_layer, verbose=False):
        self.model = model
        self.device = device
        self.hooks = FeatureHooks(verbose)
        self.activations_handle = last_conv_layer.register_forward_hook(self.hooks.save_activations)
        self.gradients_handle   = last_conv_layer.register_full_backward_hook(self.hooks.save_gradients)

    def run(self,  instance, returned_cam_for_label=None):
        self.model.eval()
        original_length = len(instance)

        instance     = Variable(torch.tensor(instance.reshape((1, 1, original_length))).float().to(self.device), requires_grad=True)
        output       = self.model(instance)
        label_pred   = output.argmax(dim=1)

        if returned_cam_for_label is None:
            output[:, returned_cam_for_label].backward()
        else:
            output[:, label_pred].backward()

        pooled_gradients = torch.mean(self.hooks.gradients, dim=[0, 2])
        for i in range(self.hooks.activations.size(1)):
            self.hooks.activations[:, i, :] *= pooled_gradients[i]

        cam = torch.mean(self.hooks.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam /= torch.max(cam)
        
        self._remove_hooks()

        return cam.detach().cpu().numpy(), label_pred.detach().cpu().squeeze().numpy()

    def _remove_hooks(self):
        self.activations_handle.remove()
        self.gradients_handle.remove()
