import torch
from typing import List, Tuple

class ResultPicker(object):
    @staticmethod
    def dump(result):
        if isinstance(result, torch.Tensor):
            return result.flatten(), result.shape
        elif isinstance(result, Tuple) or isinstance(result, List):
            result_tensor = torch.zeros(0)
            result_structure = []
            for res in result:
                
                #### for sd 3 ###
                if res==None:
                    res = result[1]
                #################

                res_tensor, res_structure = ResultPicker.dump(res)
                result_tensor = result_tensor.to(res_tensor.device)
                result_tensor = torch.cat((result_tensor, res_tensor))
                result_tensor = result_tensor.to(res_tensor.device, dtype=res_tensor.dtype)
                result_structure.append(res_structure)
            return result_tensor, result_structure
        
    @staticmethod
    def load(tensor, tensor_structure):
        if isinstance(tensor_structure, torch.Size):
            return tensor.view(tensor_structure)
        elif isinstance(tensor_structure, list):
            results = []
            start = 0
            for structure in tensor_structure:
                if isinstance(structure, torch.Size):
                    numel = torch.tensor(structure).prod().item()
                    end = start + numel
                    result = ResultPicker.load(tensor[start:end], structure)
                    results.append(result)
                    start = end
                else:
                    result = ResultPicker.load(tensor[start:], structure)
                    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
                        start += result[1]
                        results.append(result[0])
                    else:
                        results.append(result)
            return tuple(results)  # Return the reconstructed tuple
        
    @staticmethod
    def get_result_structure(result):
        if isinstance(result, torch.Tensor):
            return result.shape
        elif isinstance(result, tuple) or isinstance(result, list):
            result_structure = []
            for res in result:
                result_structure.append(ResultPicker.get_result_structure(res))
            return result_structure
        else:
            raise ValueError("Result structure not supported")
