""" This script contains the implementation of the MAML algorithms designed by 
Chelsea Finn et al. (https://arxiv.org/pdf/1703.03400).
Terminology:
------------
Support set : a set of training examples 
    (inputs+labels: iterable of (img, label) pairs)
Query set : a set of test examples 
    (inputs +labels : iterable of (img, label) pairs )
Task/Dataset : Support set + Query set.
Meta-train set: a set of datasets for meta-training
Meta-test set: a set of datasets for meta-evaluation
Meta-batch size: Number of tasks to consider for a meta-iteration
"""
import time 
import copy 
import logging
import datetime
import pickle
import numpy as np
import os 

import gin
import torch 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python import debug as tf_debug
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Conv2D

from turtle_code import Turtle
from helper import ConvX
from metadl.api.api import MetaLearner, Learner, Predictor
from utils import create_grads_shell, reset_grads, app_custom_grads

# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
def set_device():
    """Automatically chooses the right device to run pytorch on

    Returns:
        str: device identifier which is best suited (most free GPU, or CPU in case GPUs are unavailable)
    """
    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = int(np.argmax(memory_available))
        print("trying gpu_id", gpu_id)
        torch.cuda.set_device(gpu_id)
        dev = torch.cuda.current_device()
    else:
        dev = "cpu"
    return dev


TURTLE_CONF = {
    "opt_fn": torch.optim.Adam,
    "T": 5, 
    "T_val": 5,
    "T_test": 5,
    "lr": 0.001,
    "act": torch.nn.ReLU(),
    "beta": 0.9,
    "meta_batch_size": 2,
    "time_input": False,
    "decouple": None,
    "input_type": "raw_grads",
    "second_order": True,
    "layer_wise": False,
    "param_lr": True,
    "history": "grads",
    "grad_clip": 10,
    "dev": set_device(),
    "batching_eps": False,
    "test_adam": False,
    "baselearner_fn": ConvX,
    "layers": [20,20,20,20,20,1]
}

@gin.configurable
class MyMetaLearner(MetaLearner):
    def __init__(self,
                meta_iterations,
                meta_batch_size,
                support_batch_size,
                query_batch_size,
                img_size,
                N_ways):
        """
        Args:
            meta_iterations : number of meta-iterations to perform, i.e. the 
            number of times the meta-learner's weights are updated.
            
            meta_batch_size : The number of (learner, task) pairs that are used
            to produce the meta-gradients, used to update the meta-learner's 
            weights after each meta-iteration.

            support_batch_size : The batch size for the support set.
            query_batch_size : The batch size for the query set.
            img_size : Integer, images are considered to be 
                        (img_size, img_size, 3)
        """
        super().__init__()
        self.meta_iterations = meta_iterations
        self.meta_batch_size = meta_batch_size
        self.support_batch_size = support_batch_size
        self.query_batch_size = query_batch_size
        self.img_size = img_size
        self.N_ways = N_ways
        self.device = TURTLE_CONF["dev"]

        self.baselearner_args = {
            "dev": self.device,
            "train_classes": self.N_ways,
            "eval_classes": self.N_ways,
            "criterion":nn.CrossEntropyLoss(),
            "num_blocks": 4,
            "img_size": (1, 3, self.img_size, self.img_size)
        }

        TURTLE_CONF["baselearner_args"] = self.baselearner_args
        TURTLE_CONF["train_batch_size"] = self.N_ways * self.support_batch_size
        TURTLE_CONF["test_batch_size"] = self.N_ways * self.query_batch_size
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)
        self.turtle = Turtle(**TURTLE_CONF)


    def dataloader(self, dataset_episodic):
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        # 2
        
        def data_loader(n_batches):
            for i, (e, _) in enumerate(dataset_episodic):
                if i == n_batches:
                    break
                logging.info('e shape: {}'.format(len(e)))
                yield (to_torch_imgs(e[0]), to_torch_labels(e[1]),
                    to_torch_imgs(e[3]), to_torch_labels(e[4]))

        datal = data_loader(n_batches=1)
        for i, batch in enumerate(datal):
            #3
            data_support, labels_support, data_query, labels_query = [x.to(device=self.device) for x in batch]
            logging.info('Supp imgs: {} | Supp labs : {} | Query imgs : {} | Query labs \n \n'.format(data_support.shape, labels_support.shape, data_query.shape, labels_query.shape))


    def process_task(self, batch):
        """
        batch : [sup_imgs, sup_labs, sup_tidx, qry_imgs, qry_labs, qry_tidx]
        sup_imgs : [batch_idx, nbr_imgs, H, W, C]
        """
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 1, 4, 2, 3)))
        return (to_torch_imgs(batch[0]), to_torch_labels(batch[1]),
                    to_torch_imgs(batch[3]), to_torch_labels(batch[4]))

    def meta_fit(self, meta_dataset_generator):
        """ Encapsulates the meta-learning procedure. In the fo-MAML case, 
        the meta-learner's weights update. 

        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        Returns:
            A Learner object initialized with the meta-learner's weights.
        """
        # Load dataset in db
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline
        meta_train_dataset = meta_train_dataset.batch(1)
        meta_valid_dataset = meta_valid_dataset.batch(1)
        mtrain_iterator = meta_train_dataset.__iter__()
        mval_iterator = meta_valid_dataset.__iter__()


        log = []
        start_time = time.time()

        for meta_iter in range(self.meta_iterations):
            if meta_iter % 2500 == 0: 
                scores = []
                # Perform meta-validation and save model state
                for val_iter in range(2):
                    vbatch = next(mval_iterator)[0]
                    vbatch = self.process_task(vbatch)
                    vx_spt, vy_spt, vx_qry, vy_qry = [x[0].to(device=self.device) for x in vbatch]
                    vacc = self.turtle.evaluate(vx_spt, vy_spt, vx_qry, vy_qry)
                    scores.append(vacc)
                val_accuracy = np.mean(scores)
                iter_time = time.time() - start_time
                logging.info(f'[Meta-iteration {meta_iter}] Val acc: {val_accuracy:.3f} | Time: {iter_time:.2f}')
                self.turtle.store_file(os.path.join('trained_models/feedback/turtle/models', f'{meta_iter}.pkl'))
            
            batch = next(mtrain_iterator)[0]
            batch = self.process_task(batch)
            x_spt, y_spt, x_qry, y_qry = [x[0].to(device=self.device) for x in batch]
            self.turtle.train(x_spt, y_spt, x_qry, y_qry)
            break

        return MyLearner(self.turtle, self.device)

@gin.configurable
class MyLearner(Learner):
    """ In the case of fo-MAML, encapsulates a neural network and its training 
    methods.
    """
    def __init__(self, 
                meta_iterations,
                meta_batch_size,
                support_batch_size,
                query_batch_size,
                img_size,
                N_ways,
                turtle=None,
                device=None):
        """
        Args:
            neural_net : a keras.Sequential object. A neural network model to 
                        copy as Learner.
            num_epochs : Integer, the number of epochs to consider for the 
                        training on support examples.
            lr : Float, the learning rate associated to the learning procedure
                (Adaptation).
            img_size : Integer, images are considered to be 
                        (img_size,img_size,3)
        """
        super().__init__()
        self.meta_iterations = meta_iterations
        self.meta_batch_size = meta_batch_size
        self.support_batch_size = support_batch_size
        self.query_batch_size = query_batch_size
        self.img_size = img_size
        self.N_ways = N_ways
        
        
        if turtle == None:
            self.device = set_device()
            
            self.baselearner_args = {
                "dev": self.device,
                "train_classes": self.N_ways,
                "eval_classes": self.N_ways,
                "criterion":nn.CrossEntropyLoss(),
                "num_blocks": 4,
                "img_size": (1, 3, self.img_size, self.img_size)
            }

            TURTLE_CONF["dev"] = self.device
            TURTLE_CONF["baselearner_args"] = self.baselearner_args
            TURTLE_CONF["train_batch_size"] = self.N_ways * self.support_batch_size
            TURTLE_CONF["test_batch_size"] = self.N_ways * self.query_batch_size
            self.turtle = Turtle(**TURTLE_CONF)
        else:
            self.turtle = turtle
            self.device = self.turtle.dev

    # def __call__(self, imgs):
    #     return self.learner(imgs)

    def process_task(self, images, labels):
        """
        batch : [sup_imgs, sup_labs, sup_tidx, qry_imgs, qry_labs, qry_tidx]
        sup_imgs : [batch_idx, nbr_imgs, H, W, C]
        """
        to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        return to_torch_imgs(images), to_torch_labels(labels)

    def fit(self, dataset_train):
        """ The learner's fit function over the train set of a task.

        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the training 
                            examples (support set).
        Returns:
            predictor : An instance of MyPredictor that is initilialized with 
                the fine-tuned learner's weights in this case.
        """
        for images, labels in dataset_train:
            images, labels = self.process_task(images, labels)
            images, labels = images.to(self.device), labels.to(self.device)
            network, parameters = self.turtle.evaluate(images, labels, None, None, return_network=True)
            predictor = MyPredictor(network, parameters)
            break
        return predictor

    def load(self, model_dir):
        """Loads the learner model from a pickle file.

        Args:
            model_dir: the directory name in which the participant's code and 
                their saved/serialized learner are stored.
        """

        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))

        ckpt_path = os.path.join(model_dir, 'learner.pt')
        self.turtle.read_file(ckpt_path)
        
    def save(self, model_dir):
        """Saves the learner model into a pickle file.

        Args:
            model_dir: the directory name from which the participant's code and 
                their saved/serialized learner are loaded.
        """

        if(os.path.isdir(model_dir) != True):
            os.mkdir(model_dir)

        ckpt_file = os.path.join(model_dir, 'learner.pt')
        self.turtle.store_file(ckpt_file)
        
####### Predictor ########
@gin.configurable
class MyPredictor(Predictor):
    """ The predictor is meant to predict labels of the query examples at 
    meta-test time.
    """
    def __init__(self,
                 network,
                 parameters):
        """
        Args:
            learner : a MyLearner object that encapsulates the fine-tuned 
                neural network.
        """
        super().__init__()
        self.network = network
        self.parameters = [x.clone().detach() for x in parameters]
        self.device = self.parameters[0].device

    def process_imgs(self, images):
        to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
        return to_torch_imgs(images)

    def predict(self, dataset_test):
        """ Predicts labels of the query set examples associated to a task.
        Note that the query set is a tf.data.Dataset containing 50 examples for
        the Omniglot dataset.

        Args: 
            dataset_test : a tf.data.Dataset object. An iterator over the 
                unlabelled query examples.
        Returns:
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Sparse Categorical Accuracy to evaluate the predictions. Valid 
                tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.

        Note : In the challenge N_ways = 5 at meta-test time.
        """
        self.network.eval()
        for images in dataset_test:
            images = self.process_imgs(images[0]).to(self.device)
            qry_logits = self.network.forward_weights(images, self.parameters).detach()
        return qry_logits


