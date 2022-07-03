import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC


class Preparation:
    
    @staticmethod
    def get_sample_counts(df, class_names):
        total_count = df.shape[0]
        labels = df[class_names].values
        positive_counts = np.sum(labels, axis=0)
        class_positive_counts = dict(zip(class_names, positive_counts))
        return total_count, class_positive_counts

    def steps(self, df, class_names, batch_size):
        dev_counts, dev_pos_counts = self.get_sample_counts(df, class_names)
        steps = int(np.ceil(dev_counts / batch_size))
        return steps

    @staticmethod
    def calculating_class_weights(y_true):
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', classes=np.unique(y_true[:, i]), y=y_true[:, i])
        return weights

    @staticmethod
    def get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            return K.mean(
                (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * K.binary_crossentropy(y_true, y_pred),
                axis=-1)

        return weighted_loss

    @staticmethod
    def metrics_define(num_classes):
        metrics_all = ['accuracy',
                       AUC(curve='PR', multi_label=True, num_labels=num_classes, name='auc_pr'),
                       AUC(multi_label=True, num_labels=num_classes, name='auc_roc'),
                       ]
        return metrics_all
