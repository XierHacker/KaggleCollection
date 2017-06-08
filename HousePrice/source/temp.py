import numpy as np
import pandas as pd

dict={"a":range(8),"b":range(4,12)}

frame=pd.DataFrame(data=dict)
print(frame)

frame.ix[(frame["a"]>2)&(frame["a"]<5),"a"]=100
print(frame)