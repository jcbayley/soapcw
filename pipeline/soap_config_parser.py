import configparser
import json
import regex
import os

class SOAPConfig():

    def __init__(self, config_file):
        cfg = configparser.ConfigParser()
        cfg.read(config_file)

        self.config_file = os.path.abspath(config_file)
        self.float_list = ["band_starts","band_ends","band_widths"]
        self.int_list = ["strides","fc_layers","img_dim"]
        self.string_list = ["load_directory","save_options"]
        self.tuple_list = ["conv_layers"]
        self.floats = ["band_load_size", "snr_width_line", "snr_width_signal", "prob_line", "left_right_prob", "det1_prob", "det2_prob","snrmin","snrmax","learning_rate","data_load_size"]
        self.ints = ["memory", "request_disk", "n_jobs", "n_summed_sfts","n_epochs"]
        self.bools = ["resize_image", "overwrite_files"]

        self.config = self.parse_config(cfg)

    def __getitem__(self, key):
        return self.config[key]

    def load_list(self, val, partype):
        if "," in val:
            val = val.strip("[").strip("]").strip("(").strip(")").split(",")
        else:
            val = [val.strip("[").strip("]").strip("(").strip(")").strip(",")]
        
        if partype == "float":
            val = [float(v) for v in val]
        elif partype == "int":
            val = [int(v) for v in val]
        elif partype == "string":
            val = [v.replace('"','').replace(' ','') for v in val]
        else:
            raise Exception(f"Type {partype} not supported")

        return val

    def get_bool(self, val):
        if val in ["false", "False", False, 0]:
            return False
        elif val in ["true", "True", True, 1]:
            return True
        else:
            raise Exception("Value not of Bool type")
            
    def parse_config(self, cfg):

        parsed_dict = {}
        for key, val in cfg.items():
            parsed_dict[key] = {}
            for key2, val2 in val.items():
                # if comma then part of a list
                if key2 in self.float_list:
                    parsed_dict[key][key2] = self.load_list(val2, "float")
                elif key2 in self.int_list:
                    parsed_dict[key][key2] = self.load_list(val2, "int")
                elif key2 in self.string_list:
                    parsed_dict[key][key2] = self.load_list(val2, "string")
                elif key2 in self.floats:
                    parsed_dict[key][key2] = float(val2)
                elif key2 in self.ints:
                    parsed_dict[key][key2] = int(val2)
                elif key2 in self.bools:
                    parsed_dict[key][key2] = self.get_bool(val2)
                elif key2 in self.tuple_list:
                    temp_out = []
                    for mod in regex.split(r"\s*,\s*(?![^(]*\))", val2.strip("[").strip("]")):
                        if mod == "":
                            continue
                        out_tuple = tuple([int(vl) for vl in mod.strip("\n").strip("(").strip(")").split(",")])
    
                        temp_out.append(out_tuple)

                    parsed_dict[key][key2] = temp_out

                else:
                    if val2 in ["none", "None"]:
                        parsed_dict[key][key2] = None
                    else:
                        parsed_dict[key][key2] = val2.replace('"','')
                
        return parsed_dict