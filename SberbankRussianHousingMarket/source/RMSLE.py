import numpy as np
def loss(y_true,y_predict):
    if y_true.shape[0]!=y_predict.shape[0]:
        print("shape error!!!!")
        return None
    else:
        sub=np.log(y_predict+1)-np.log(y_true+1)
        squre=sub*sub
        mean=(squre.sum())/(y_true.shape[0])
        error=np.sqrt(mean)
        return error



