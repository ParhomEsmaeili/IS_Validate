def front_back_processor():
    '''
    This class serves as the interaction point between a pseudo-front end and the back-end which performs inference. Handles operations such as:

    Checking whether the back-end generated segmentations are sent back in the native/pseudo-ui image domain and are valid!!, 
    saving the outputs, processing the outputs into the required format for prompt generation and interaction state handling.
    
    '''