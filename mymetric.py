class mymetric(tf.keras.metrics.Metric):
    # custom metric 
    # to calculate the precision for 3-classification (label = 2)
    def __init__(self, name="precision", **kwargs):
        super(mymetric, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name="prec",initializer="zeros")
        self.positives = self.add_weight(name="pos",initializer="zeros")
        self.true_positives = self.add_weight(name="true_pos",initializer="zeros")
        self.times = self.add_weight(name="times",initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        y_true = tf.reshape(tf.argmax(y_true, axis=1), shape=(-1, 1))
        pred2_true = tf.cast(y_true[y_pred == 2], "int32")
        amt = tf.cast(tf.size(pred2_true), "float32")
        self.positives.assign_add(amt)
        right = tf.cast(pred2_true[pred2_true == 2], "int32")
        right = tf.cast(tf.size(right), "float32")
        self.true_positives.assign_add(right)
        self.times.assign_add(1)
    def result(self):
        return (self.true_positives,self.positives,self.times)
    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.precision.assign(0.0)
        self.positives.assign(0)
        self.true_positives.assign(0)
        self.times.assign(0)  
