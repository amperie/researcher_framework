import torch
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase
from neuralsignal.core.modules.feature_sets.feature_utils\
    import is_layer_string_match_in_list
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import login


class FeatureSetLogitLens(FeatureSetBase):

    def _load_model(self, model_name: str, hf_token: str):

        login(hf_token)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_name,
                                device_map=self.dev_map,
                                # load_in_8bit=True,
                                )
        self.unembed = self.model.lm_head

    def __init__(self, config: dict):
        """
        cfg must contain the following configuration:
        model_name: model from where to get the unembedding matrix
        hf_token: huggingface token
        layers_to_process: list of layer name string matches to process
        """
        super().__init__(config)
        if "dev_map" in config:
            dev_map = config["dev_map"]
        elif torch.cuda.is_available():
            dev_map = "cuda:0"
        else:
            dev_map = "cpu"
        self.dev_map = dev_map

        if "unembed_layer" in config:
            self.unembed = config["unembed_layer"]
        else:
            self._load_model(config['model_name'], config['hf_token'])

    def get_feature_set_name(self) -> str:
        return "logit-lens"

    def process_feature_set(self, scan: dict):
        """
        Process the zone sizes by layer for a given scan and configuration.

        Args:
            scan (dict): A dictionary containing the scan data.

        Returns: Depending on the config parameter output_format outputs are
        "name_and_value_columns", "tensor_dict" and "pandas"
        tensor_dict:
            tuple: Tensor dictionary, Dictionary of zone sizes
        name_and_value_columns:
            tuple: column names, column values
        pandas:
            DataFrame: contains the features and columns in a pandas df

        """
        self.scan = scan
        cols = []
        vals = []
        idx = 0

        layers_to_process = self.config['layers_to_process']
        for lyr in scan['outputs'].keys():
            lyr_name = scan['layer_id_to_name'][lyr]
            if is_layer_string_match_in_list(lyr_name, layers_to_process):
                # Layer is in the list to process
                # Get the logits from it by feeding it into the unembed matrix
                t = scan['outputs'][lyr]
                t = t.to(self.dev_map)
                logits = self.unembed.forward(t)
                cols.append(self.make_column_name(f"std_{lyr_name}_{idx}"))
                vals.append(torch.std(logits).item())
                cols.append(self.make_column_name(f"mean_{lyr_name}_{idx}"))
                vals.append(torch.mean(logits).item())
                idx += 1

        # Return the right format results
        output_format = self.config['output_format']
        if output_format == "name_and_value_columns":
            return (cols, vals)
        elif output_format == "tensor_dict":
            return None
        elif output_format == "pandas":
            return pd.DataFrame([vals], columns=cols)
        else:
            raise ValueError(
                "output_format must be one of 'name_and_value_columns', "
                "'tensor_dict' or 'pandas'"
                )
