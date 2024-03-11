import os
import ast
import copy
import logging
import configparser as cp
from types import SimpleNamespace
from datetime import datetime as dt
from typing import Dict, List, Optional


class ConfigParser:
    """Configuration file parser class. Contains methods for both reading and saving
    configuration files. Also contains methods for meta data saving after training.
    """

    root: str = "cfg"
    train_root: str = os.path.join(root, "train")
    model_root: str = os.path.join(root, "models")
    common_path: str = os.path.join(root, "common.ini")

    saved_conf_fname: str = "settings.ini"
    nas_token: str = "{#NAS}"
    exluded_sections: List[str] = [] # ["path", "database", "notifier", "neptune"]  # These sections are not saved 

    @classmethod
    def get(cls, task: str) -> SimpleNamespace:
        """
        Read configuration file
        Return the merged configuration instance.
        """
       
        task_ini = os.path.join(cls.train_root, task + ".ini")
        conf = cls.read(cls.common_path, task_ini)

        return conf
    
    @classmethod
    def prepare_conf(cls, args: dict, modifiers=None) -> SimpleNamespace:
        """
        Create configuration object from file and command line arguments.
        """

        conf = ConfigParser.get(args.task)
        conf = ConfigParser.expand(conf, args)
        return conf


    @classmethod
    def expand(cls, conf: SimpleNamespace, info: Dict) -> SimpleNamespace:
        """
        Expand the configuration file by adding the path to the division and the
        path to the root.
        """
        if hasattr(conf, "dynamic"):
            conf.dynamic = SimpleNamespace(**{**vars(conf.dynamic), **info})
        else:
            conf.dynamic = SimpleNamespace(**info.__dict__)
        return conf


    @classmethod
    def save(cls, conf: SimpleNamespace, file_path: str) -> None:
        """
        Save configuration SimpleNamespace to file.
        """
        config = cp.ConfigParser()
        for section, variables in conf.__dict__.items():
            config[section] = {}
            for name, value in variables.__dict__.items():
                data = f"'{value}'" if isinstance(value, str) else str(value)
                config[section][name] = data

        # Delete excluded sections
        for section in cls.exluded_sections:
            try:
                if section == "notifier":
                    if section in config._sections:
                        del config._sections[section]
                    else:
                        print(f'Skipping section "{section}" because it is optional and does not exist in the configuration.')
                else:
                    del config._sections[section]
            except KeyError:
                raise KeyError(f'Unknown section name in excluded sections: "{section}"') from None
        with open(file_path, "w") as f:
            config.write(f)

    # Private methods

    @classmethod
    def read(cls, *config_files: str) -> SimpleNamespace:
        """
        Read in all the given configuration files and return a SimpleNamespace instance.
        """
        file_conf = cp.ConfigParser()

        # Read in config files
        for ini_file in config_files:
            if not os.path.isfile(ini_file):
                raise ValueError(f"{ini_file} file does not exist")
            file_conf.read(ini_file, encoding="utf8")

        # convert to namespace
        conf_dict = {}
        for section_name in file_conf.sections():
            d = {}
            for key, val in file_conf.items(section_name):
                try:
                    d[key] = ast.literal_eval(val)
                except Exception as e:
                    logging.error(e)
                    logging.error(f"Malfromed config: {section_name}/{key}")
            item = SimpleNamespace(**d)
            conf_dict[section_name] = item
        conf = SimpleNamespace(**conf_dict)

        return conf

