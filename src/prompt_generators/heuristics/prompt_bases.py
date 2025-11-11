#This file contains the base classes for each of the prompt types. It is intended to provide each prompt type 
#with the capability to implement methods intended for inheritance within downstream mixture models
# for a given prompting type. TODO: Integrate this with the mixture classes properly in a later refactor.

class PointBase:
    pass 

class ScribbleBase:
    pass 

class BboxBase:
    pass 
class LassoBase:
    pass 