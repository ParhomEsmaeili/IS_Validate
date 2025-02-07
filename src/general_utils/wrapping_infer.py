import copy
import logging
import os
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
# from monai.data import decollate_batch
# from monai.inferers import Inferer, SimpleInferer, SlidingWindowInferer
# from monai.utils import deprecated

# from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
# from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
# from monailabel.interfaces.utils.transform import dump_data, run_transforms
# from monailabel.transform.cache import CacheTransformDatad
# from monailabel.transform.writer import ClassificationWriter, DetectionWriter, Writer
# from monailabel.utils.others.generic import device_list, device_map, name_to_device

class CallBackTypes(str, Enum):
    INFERER = "INFERER"
    WRITER = "WRITER"

def __call__(
        self, request, callbacks: Union[Dict[CallBackTypes, Any], None] = None
    ) -> Union[Dict, Tuple[str, Dict[str, Any]]]:
        """
        It provides basic implementation to run the following in order
            - Run Pre Transforms
            - Run Inferer
            - Run Invert Transforms
            - Run Post Transforms
            - Run Writer to save the label mask and result params

        You can provide callbacks which can be useful while writing pipelines to consume intermediate outputs
        Callback function should consume data and return data (modified/updated) e.g. `def my_cb(data): return data`

        Returns: Label (File Path) and Result Params (JSON)
        """
        begin = time.time()
        req = copy.deepcopy(self._config)
        req.update(request)

        # device
        device = name_to_device(req.get("device", "cuda"))
        req["device"] = device

        logger.setLevel(req.get("logging", "INFO").upper())
        if req.get("image") is not None and isinstance(req.get("image"), str):
            logger.info(f"Infer Request (final): {req}")
            data = copy.deepcopy(req)
            data.update({"image_path": req.get("image")})
        else:
            dump_data(req, logger.level)
            data = req

        # callbacks useful in case of pipeliens to consume intermediate output from each of the following stages
        # callback function should consume data and returns data (modified/updated)
        callbacks = callbacks if callbacks else {}
        #This is the inferer implemented by the user application which is fully end-to-end.
        callback_run_inferer = callbacks.get(CallBackTypes.INFERER)
        #This is the writer implemented for saving the segmentation and interaction states.
        callback_writer = callbacks.get(CallBackTypes.WRITER)

        start = time.time()
        pre_transforms = self.pre_transforms(data)
        data = self.run_pre_transforms(data, pre_transforms)
        if callback_run_pre_transforms:
            data = callback_run_pre_transforms(data)
        latency_pre = time.time() - start

        start = time.time()
        
        
        data = self.run_inferer(data, device=device)

        if callback_run_inferer:
            data = callback_run_inferer(data)
        latency_inferer = time.time() - start

        start = time.time()
        data = self.run_invert_transforms(data, pre_transforms, self.inverse_transforms(data))
        if callback_run_invert_transforms:
            data = callback_run_invert_transforms(data)
        latency_invert = time.time() - start

        start = time.time()
        data = self.run_post_transforms(data, self.post_transforms(data))
        if callback_run_post_transforms:
            data = callback_run_post_transforms(data)
        latency_post = time.time() - start

        if self.skip_writer:
            return dict(data)

        start = time.time()
        result_file_name, result_json = self.writer(data)
        if callback_writer:
            data = callback_writer(data)
        latency_write = time.time() - start

        latency_total = time.time() - begin
        logger.info(
            "++ Latencies => Total: {:.4f}; "
            "Pre: {:.4f}; Inferer: {:.4f}; Invert: {:.4f}; Post: {:.4f}; Write: {:.4f}".format(
                latency_total,
                latency_pre,
                latency_inferer,
                latency_invert,
                latency_post,
                latency_write,
            )
        )

        result_json["label_names"] = self.labels
        result_json["latencies"] = {
            "pre": round(latency_pre, 2),
            "infer": round(latency_inferer, 2),
            "invert": round(latency_invert, 2),
            "post": round(latency_post, 2),
            "write": round(latency_write, 2),
            "total": round(latency_total, 2),
            "transform": data.get("latencies"),
        }

        # Add Centroids to the result json to consume in OHIF v3
        centroids = data.get("centroids", None)
        if centroids is not None:
            centroids_dict = dict()
            for c in centroids:
                all_items = list(c.items())
                centroids_dict[all_items[0][0]] = [str(i) for i in all_items[0][1]]  # making it json compatible
            result_json["centroids"] = centroids_dict
        else:
            result_json["centroids"] = dict()

        if result_file_name is not None and isinstance(result_file_name, str):
            logger.info(f"Result File: {result_file_name}")
        logger.info(f"Result Json Keys: {list(result_json.keys())}")
        return result_file_name, result_json
