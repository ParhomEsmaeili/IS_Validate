import tempfile
import json
import os 
import shutil 

if __name__ == "__main__":
    print('HALT, stop and check the tempfile here from prior iterations')

    dummy_dict = {'1':1, '2':2, '3':3, '4':4, '5':5}

    tmp_dir = os.path.join(os.path.abspath(os.path.expanduser('~')), 'tmp_dir')
    os.makedirs(tmp_dir, exist_ok=True)
    
    tempfile_obj = tempfile.NamedTemporaryFile(suffix='.json', dir = tmp_dir, delete=False)
    tempfile_path = tempfile_obj.name

    with open(tempfile_path, 'w') as f:
        json.dump(dummy_dict, f)

    with open(tempfile_path, 'r') as f:
        print(json.load(f))
    tempfile_obj.close()

    os.remove(tempfile_path)
    shutil.rmtree(tmp_dir)
    print('HALT: stop and check the tempfile here')