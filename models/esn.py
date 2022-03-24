from dependency.pyESN import ESN
import numpy as np

class ESNModel():
    def __init__(self, n_reservoir:int=200):
        self.model = ESN(
            n_inputs=1,
            n_outputs=1,
            n_reservoir=n_reservoir,
            sparsity=0.2,
            #random_state=23,
            spectral_radius=1.2,
            noise=0.0005
        )

    def train_and_predict(self, train_len:int, future:int, future_total:int, data:np.array, offset:int=0) -> np.array:
        predictions = np.zeros(future_total)
        for i in range(0,future_total,future):
            self.model.fit(np.ones(train_len),data[i+offset:train_len+i+offset])
            pred = self.model.predict(np.ones(future))
            predictions[i:i+future] = pred[:,0]
            print('Iteration: ' + str(int(i/future + 1)) + '/' + str(int(future_total/future)))
        return np.array(predictions)