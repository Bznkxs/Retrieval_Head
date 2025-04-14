import json
import os


class Config:

    @staticmethod
    def get_config(config_root, config_name):
        if not config_name.endswith('.json'):
            config_name += '.json'
        if not os.path.exists(config_root):
            os.makedirs(config_root)
        config_file_full_path = os.path.join(config_root, config_name)
        if not os.path.exists(config_file_full_path):
            with open(config_file_full_path, 'w') as config_file:
                config_file.write("{}")  # empty
        with open(config_file_full_path, 'r') as config_file:
            try:
                return json.load(config_file)
            except json.decoder.JSONDecodeError:
                with open(config_file_full_path, 'w') as config_file:
                    config_file.write("{}")  # empty
                return {}


    @staticmethod
    def write_config(config_root, config_name, config_object):
        if not config_name.endswith('.json'):
            config_name += '.json'
        if not os.path.exists(config_root):
            os.makedirs(config_root)
        config_file_full_path = os.path.join(config_root, config_name)
        with open(config_file_full_path, 'w') as config_file:
            json.dump(config_object, config_file)

    def __init__(self, name, config_root='.manager_config'):
        self.name = name
        self.config_root = config_root
        self._config = None

    def __getitem__(self, key):
        if self._config is None:
            self._config = self.get_config(self.config_root, self.name)
        return self._config[key]

    def get(self, key, default=None):
        if self._config is None:
            self._config = self.get_config(self.config_root, self.name)
        return self._config.get(key, default)

    def pop(self, key):
        if self._config is None:
            self._config = self.get_config(self.config_root, self.name)
        return self._config.pop(key)

    def __setitem__(self, key, value):
        if self._config is None:
            self._config = self.get_config(self.config_root, self.name)
        self._config[key] = value
        self.write_config(self.config_root, self.name, self._config)

    def sync(self):
        self.write_config(self.config_root, self.name, self._config)

    def exists(self):
        config_name = self.name
        if not config_name.endswith('.json'):
            config_name += '.json'
        return os.path.exists(os.path.join(self.config_root, config_name))

