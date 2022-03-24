from dependency.pyESN import ESN
import numpy as np

class ESNModel():
    def __init__(self, n_reservoir:int=200):
        self.model = ESN(
            n_inputs=1,
            n_outputs=1,
            n_reservoir=n_reservoir,
            sparsity=0.2,
            random_state=23,
            spectral_radius=1.2,
            noise=0.0005
        )

    def train_and_predict(self, train_len:int, future:int, future_total:int, data:np.array) -> np.array:
        predictions = np.zeros(future_total)
        for i in range(0,future_total,future):
            pred_training = self.model.fit(np.ones(train_len),data[i:train_len+i])
            pred = self.model.predict(np.ones(future))
            predictions[i:i+future] = pred[:,0]
        return predictions