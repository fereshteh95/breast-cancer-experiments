import mlflow


class PyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import tensorflow as tf
        from omegaconf import OmegaConf

        self.model_ = tf.keras.models.load_model(context.artifacts['savedmodel_path'])
        self.config = OmegaConf.load(context.artifacts['config_path'])
        # self.threshold = self.config.model_builder.threshold

    def predict(self, context, model_input):
        """Process a list of images.
        Args:
            model_input: an array of png images of shape(256, 256, 3) and range(0, 255)
        """

        from time import time

        ts = time()
        out = self.model_.predict(model_input)
        te = time()
        # labels = list(map(lambda x: map_d[x], np.argmax(out, axis=1)))

        latency = (te - ts) / len(out)

        results = []

        for i, pred in enumerate(out):
            ret = {'output': pred,
                   'latency': latency,
                   'index': i}
            results.append(ret)

        return results
