import tempfile
import json
import os 

if __name__ == "__main__":
    print('HALT, stop and check the tempfile here from prior iterations')

    dummy_dict = {'1':1, '2':2, '3':3, '4':4, '5':5}

    tempfile_obj = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    tempfile_path = tempfile_obj.name  

    with open(tempfile_path, 'w') as f:
        json.dump(dummy_dict, f)
    

    with open(tempfile_path, 'r') as f:
        print(json.load(f))
    tempfile_obj.close() 

    os.remove(tempfile_path)
    print('HALT: stop and check the tempfile here')