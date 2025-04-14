"""
Welcome to the omnipotent manage.py! This script will systematically
manage the conduction of experiments using gsm8k_rope.py on clusters
using slurm as scheduler.

# Experiment Structure Management

1. An experiment contains one or multiple tests.
2. An experiment is defined by its name and version. Two experiments are the same only if both the names and versions are the same.
3. A test always bears the name of the experiment it belongs to. The test has its specifications.
4. The tester, defined in `gsm8k_rope.py`, deals with one test.

# Experiment File Management

1. A tester operates only in one test directory.
2. One test directory stores one test.
3. One experiment directory stores one experiment, which may contain one or multiple test directories.
4. The Results Directory (by default results/rope) stores experiment and test directories.
5. Only test directories that are created prior to the creation of THIS FILE, or those
   generated directly by running python gsm8k_rope.py, are allowed to exist directly in The Results Directory.
6. Test directories directly contained in The Results Directory must follow certain rules as defined in
   the code below with their naming conventions.
7. Test directories created after the creation of THIS FILE must contain a file named test_info.json with its specs.
8. Experiment directories are identified by containing a file named experiment_info.json that contains the information
   and naming convention of the experiment and its tests.

# Specs
1. A test specs is a dict that describes one or more specifications for a test (a test spec).
2. Each test spec in the test specs has one name and one value.
3. An experiment specs is a dict that describes the collection of test specs involved in an experiment.
4. Each experiment specification has one name and one range of choices.
5. The value of a test specification can only occur in the range of the corresponding experiment specification.
6. The running test specs is a less important test specs. It defines dynamic information needed for actually running
   the test (url, api_key, start/end index of problem list, ...). These specs are usually dynamically provided or fixed
   throughout the experiment.
#

"""
import argparse
import copy
import glob
import json
import os
import queue
import threading
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Optional

from gsm8k_rope import FillerTester, HoleFillingTester, debug, info
import gsm8k_rope
from managing.config import Config
from managing.job_on_node import BackendSubmitter
from managing.slurm_related import parse_sbatch_file
from managing.terminal_formatting import clen, printable_table, refresh_string


class ExperimentSpec:
    def __init__(self, spec_name, spec_type, spec_range=None, spec_choices=None):
        """
        spec_type: None/"object"
        spec_range: int only; a list of 1, 2, or 3 integers, in the format of python range
        spec_choices: overrides spec_range
        """
        self.spec_name = spec_name
        self.spec_type = spec_type
        self.spec_range = spec_range
        self.spec_choices = spec_choices
        assert spec_range is None or spec_choices is None

    @staticmethod
    def from_dict(spec_obj, name=None):
        if name is None:
            assert spec_obj.get('name') is not None
            name = spec_obj.get('name')
        if spec_obj is None:
            return ExperimentSpec(name, "object", None, [None])
        if isinstance(spec_obj, dict):
            return ExperimentSpec(name, spec_obj.get('type', "object"), spec_obj.get('range'), spec_obj.get('choices'))
        if isinstance(spec_obj, list):
            if len(spec_obj) == 0:
                return ExperimentSpec(name, None)
            return ExperimentSpec(name, "object", spec_choices=spec_obj)
        if isinstance(spec_obj, str) or isinstance(spec_obj, int) or isinstance(spec_obj, float):
            return ExperimentSpec(name, "object", spec_choices=[spec_obj])

    def to_dict(self, contains_name=False):
        if not contains_name:
            return {
                "type": self.spec_type,
                "range": self.spec_range,
                "choices": self.spec_choices
            }
        return {
            "name": self.spec_name,
            "type": self.spec_type,
            "range": self.spec_range,
            "choices": self.spec_choices
        }

    def expand(self):
        if self.spec_type is None:
            return None
        if self.spec_choices is not None:
            return self.spec_choices
        if self.spec_range is None:
            return None
        return list(range(*self.spec_range))


class ExperimentSpecs:
    def __init__(self, specs: Dict[str, ExperimentSpec]):
        self.specs = specs
        self.specs_keys = list(self.specs.keys())
        self.specs_keys.sort()

    @staticmethod
    def from_pure_dict(specs_dict):
        specs = {name: ExperimentSpec.from_dict(specs_dict[name], name) for name in specs_dict}
        return ExperimentSpecs(specs)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.specs.items()}

    def _dfs(self, spec_idx, current_spec):
        if spec_idx == len(self.specs):
            return [TestSpecs.from_pure_dict(current_spec)]
        key = self.specs_keys[spec_idx]
        expansion = self.specs[key].expand()
        if expansion is None:  # skip the none specs
            return self._dfs(spec_idx + 1, current_spec)
        possible_specs = []
        future_spec = current_spec.copy()
        for spec_setup in expansion:
            future_spec[key] = spec_setup
            possible_specs.extend(self._dfs(spec_idx + 1, future_spec))
        return possible_specs

    def expand(self):
        # cartesian product for all specs
        return self._dfs(0, {})


class TestSpec(ExperimentSpec):
    def __init__(self, spec_name, spec_type, spec_value):
        """
        spec_type "dynamic" can be used as a placeholder where spec_value will be ignored.
        """
        super().__init__(spec_name, spec_type, spec_choices=[spec_value])
        self.spec_value = spec_value

    @staticmethod
    def from_dict(spec_obj, name=None):
        experiment_spec = ExperimentSpec.from_dict(spec_obj, name)
        return TestSpec.from_experiment_spec(experiment_spec)

    @staticmethod
    def from_experiment_spec(experiment_spec: ExperimentSpec):
        if isinstance(experiment_spec, TestSpec):
            return TestSpec(experiment_spec.spec_name, experiment_spec.spec_type, experiment_spec.spec_value)
        expansion = experiment_spec.expand()
        if expansion is not None:
            assert len(expansion) == 1
            return TestSpec(experiment_spec.spec_name, experiment_spec.spec_type, expansion[0])
        return TestSpec(experiment_spec.spec_name, None, None)


class TestSpecs(ExperimentSpecs):
    def __init__(self, specs: Dict[str, ExperimentSpec]):
        super().__init__(specs)
        self.specs = {k: TestSpec.from_experiment_spec(v) for k, v in specs.items()}

    def realize(self, other: "TestSpecs"):
        """
        This is very strict: you have to provide only keys that appear in the template specs (self),
        and cover all the dynamic types
        todo: this should be a function of ExperimentSpecs
        """
        ret = self.specs.copy()
        ret.update(other.specs)
        for k in ret:
            if ret[k].spec_type == "dynamic":
                return None
        return TestSpecs(ret)

    @staticmethod
    def from_pure_dict(specs_dict):
        specs = {name: TestSpec.from_dict(specs_dict[name], name) for name in specs_dict}
        return TestSpecs(specs)

    def to_dict(self):
        return {k: v.spec_value for k, v in self.specs.items()}

    def to_string(self):
        return ','.join(f"{k}={self.specs[k].spec_value}" for k in self.specs_keys)


class Hook:
    def __init__(self):
        self.unhooked_event = threading.Event()
        self.stop_event = threading.Event()
        self.unhook()
        self.auxiliary = None
        self.lock = threading.Lock()

    def is_hooked(self):
        return not self.unhooked_event.is_set()

    def unhook(self):
        self.unhooked_event.set()

    def hook(self):
        self.unhooked_event.clear()

    def stop(self):
        with self.lock:
            self.stop_event.set()

    def set_stop_event(self, stop_event):
        with self.lock:
            self.stop_event = stop_event


def path_type(path):
    if not os.path.exists(path):
        return "nonexistent"
    if not os.path.isdir(path):
        return "file"
    if os.path.exists(os.path.join(path, "test_info.json")):
        return "test_dir"
    else:
        if os.path.exists(os.path.join(path, "experiment_info.json")):
            return "experiment_dir"
        else:
            if len(os.listdir(path)) == 0:
                return "empty_dir"
            return "unknown"


def get_info_string_from_info(node_info):
    """
    a wrapper for ordering and json dumping a dict
    """
    item_list = list(node_info.items())
    item_list.sort()
    return json.dumps(item_list)


class ManagerTreeNode:
    """
    The base class for the manager hierarchy.
    Key points:
    - Directory-based: Each instance corresponds to one directory
    - In-place update: Only one instance for each directory
    - Thread-safe
    """

    config_lock = threading.Lock()

    def __init__(self, directory, config_root, node_definition, node_info):
        """
        node_definition: see get_def_from_path
        """
        self._directory = directory  # read only
        self.info_lock = threading.Lock()
        self.child_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.children = {}
        self.node_definition = node_definition
        self._copiable_node_info = node_info
        self._node_info_others = {}
        if config_root is None:
            config_root = self.directory_absolute_path()
        self.config = Config(self.config_name, config_root)

    @classmethod
    def config_name(cls):
        return "node_config"

    @classmethod
    def get_children_class(cls) -> Optional["ManagerTreeNode"]:
        return None

    @classmethod
    def get_info_from_path(cls, path):
        """
        do not include any info that is not contained in the path
        """
        raise NotImplementedError

    @classmethod
    def get_def_from_path(cls, path, parent_instance=None):
        """
        node_definition: a dict of information that DEFINES the node
        """
        raise NotImplementedError

    @classmethod
    def from_path(cls, path, parent_instance=None):
        """
        new instance from path
        """
        raise NotImplementedError

    def directory_absolute_path(self):
        return Path(self._directory).absolute()

    def is_child(self, path):
        raise NotImplementedError

    def _get_info_unsafe(self):
        ret = copy.deepcopy(self._copiable_node_info)
        ret.update(self._node_info_others)  # we cannot deepcopy this
        return ret

    def get_info(self):
        with self.info_lock:
            self._get_info_unsafe()

    def get_def(self):
        with self.info_lock:
            ret = copy.deepcopy(self.node_definition)
            return ret

    def __repr__(self):
        return f"<ManagerTreeNode Type {self.__class__.__name__} Instance Defined By {get_info_string_from_info(self.get_def())}>"

    def __hash__(self):
        return hash(repr(self))

    def get_directory(self, create_if_not_exist=True):
        directory_full_path_status = path_type(self.directory_absolute_path())
        if directory_full_path_status == "file" or "test_dir" in directory_full_path_status:
            raise FileExistsError(
                f"{self.directory_absolute_path()}, which is the designated path for the {self.__class__.__name__}, already exists and is of type {directory_full_path_status}.")
        if directory_full_path_status == "experiment_dir":
            return self.directory_absolute_path()
        if create_if_not_exist:
            with self.file_lock:
                os.makedirs(self.directory_absolute_path(), exist_ok=True)
            return self.directory_absolute_path()
        return None

    def get_path_list_for_potential_children(self):
        raise NotImplementedError

    def get_children_from_path(self, force_update=True):
        if self.get_children_class() is None:
            return {}
        with self.child_lock:
            if not force_update and len(self.children) > 0:
                # we cannot deepcopy children
                return copy.copy(self.children)
            children_backup = self.children  # pointers
            self.children = {}
            # find in The Results Directory
            potential_paths = self.get_path_list_for_potential_children()
            for path in potential_paths:
                if self.is_child(path):
                    child_def = self.get_children_class().get_def_from_path(path, self)
                    child_def_string = get_info_string_from_info(child_def)
                    if child_def_string in children_backup:
                        self.children[child_def_string] = children_backup[child_def_string]
                        self.children[child_def_string].update_self()
                    else:
                        self.children[child_def_string] = self.get_children_class().from_path(path)
            return copy.copy(self.children)

    def update_self(self):
        self.get_children_from_path()
        # update self info from path. Def is not going to change
        new_info = self.get_info_from_path(self.directory_absolute_path())
        with self.info_lock:
            self._copiable_node_info.update(new_info)
            return self._get_info_unsafe()  # we are in info lock


class Test(ManagerTreeNode):
    def __init__(self, experiment_name, experiment_version, savefile_root,
                 linking_tester_class, static_test_specs: TestSpecs,):

        linking_tester_class_str = linking_tester_class if (
            isinstance(linking_tester_class, str)) else linking_tester_class.__name__
        tester = linking_tester_class(experiment_name=experiment_name,
                                          experiment_version=experiment_version,
                                          savefile_root=savefile_root,
                                          experiment_specs_str=static_test_specs.to_string(), )
        directory = tester.find_save_path()
        super().__init__(directory, directory, {"tester_class_str": linking_tester_class_str,
                                                "experiment_name": experiment_name,
                                                "specs": static_test_specs}, None)
        # the config is managed by tester

    @classmethod
    def get_children_class(cls):
        return None

    @classmethod
    def config_name(cls):
        return "test_info"

    @classmethod
    def get_def_from_path(cls, path, parent_instance=None):
        debug(f"Trying to get info from discovered test directory {path} ...")

        config = Config(cls.config_name(), path)
        specs_dict = None
        linking_tester_class = parent_instance.linking_tester_class if parent_instance else None
        if config.exists():
            with cls.config_lock:
                specs_dict = config.get("specs")
                linking_tester_class_from_specs = globals().get(config.get("tester"))
                if linking_tester_class and linking_tester_class_from_specs:
                    assert linking_tester_class == linking_tester_class_from_specs
                linking_tester_class = linking_tester_class or linking_tester_class_from_specs
                if not specs_dict:
                    debug(f"  {path}/{cls.config_name()}.json found but there is no specs definition.")
        if specs_dict is None and parent_instance:
            specs_dict = parent_instance.linking_tester_class.parse_conventional_name(path,
                                                                                      parent_instance.experiment_name_for_tester())
        if specs_dict is None:
            specs_dict = parent_instance.parse_conventional_name(path,
                                                                 parent_instance.experiment_name_for_tester_legacy())
        if specs_dict is None:
            debug(f"  Cannot get specs from {path}. Ignoring this directory.")
            return False
        debug(f"   Got specs:", specs_dict)
        assert linking_tester_class is not None, f"Must provide linking_tester_class for test directory {path}"
        return {"tester_class_str": linking_tester_class.__name__,
                "specs": specs_dict,
                "experiment_name": parent_instance.experiment_name_for_tester()}

    @classmethod
    def get_info_from_path(cls, path):
        return {}

    @classmethod
    def from_path(cls, path, parent_instance=None):
        definition = cls.get_def_from_path(path, parent_instance)
        tester_class = globals().get(definition["tester_class_str"])
        tester_class.conventional_naming()


class Experiment:
    def __init__(self, name, version, experiment_type, linking_tester_class,
                 the_results_directory=None, naming_style=None, verbosity=False):
        """
        naming_style: None or "legacy" or "manager"
        """
        self.verbosity = verbosity
        self.name = name
        if self.name == "":
            raise ValueError("Name must not be empty!")
        self.experiment_type = experiment_type
        self.version = version

        self.linking_tester_class = globals()[linking_tester_class] if isinstance(linking_tester_class,
                                                                                  str) else linking_tester_class
        self.global_config = Config(".global")
        self.the_results_directory = the_results_directory or self.global_config.get("the_results_directory")
        if not self.the_results_directory:
            raise ValueError(f"The Results Directory is not provided for experiment {self.class_name_version_string()}")
        self.naming_style = naming_style
        self.test_directories = {}

        self.type_config = Config(experiment_type)
        self.config = Config("experiment_info", self.directory_full_path())
        self.opened_tests = {}

        self.resource_lock = threading.Lock()

        self.tester_default_specs = self.linking_tester_class.default_specs()

        # super().__init__(self.directory_name(), None, {
        #     "experiment_name": self.name,
        #     "experiment_type": self.experiment_type,
        #     "version": self.version
        # }, {
        #     "linking_tester_class": self.linking_tester_class.__name__
        # })

    def exists(self):
        existing_experiment_directory = self.get_experiment_directory(False)
        return existing_experiment_directory is not None

    def inscribe(self):
        self.get_experiment_directory()
        global_config_experiment_types = self.global_config.get("experiment_types", [])
        global_config_experiment_types_set = set(global_config_experiment_types)
        global_config_experiment_types_set.update(self.experiment_type)
        self.global_config["experiment_types"] = list(global_config_experiment_types_set)
        self.config["specs"] = self.config.get("specs", {})
        self.config["experiment_name"] = self.name
        self.config["experiment_type"] = self.experiment_type
        self.config["version"] = self.version
        self.config["linking_tester_class"] = self.linking_tester_class.__name__
        self.config["naming_style"] = self.naming_style
        self.config["running_specs"] = self.config.get("running_specs", {})

    def resource_type(self):
        # HARDCODED
        if self.get_experiment_specs().specs["model_class_name"].spec_choices.__len__() == 1:
            if self.get_experiment_specs().specs["model_class_name"].spec_choices[0] == "MegatronModel":
                return "gpu"
            else:
                return "openai_api_key"

    @staticmethod
    def from_directory(directory: str):
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return None
        if not os.path.isfile(os.path.join(directory, "experiment_info.json")):
            return None
        config = Config("experiment_info", directory)
        return Experiment(name=config["experiment_name"],
                          version=config["version"],
                          experiment_type=config["experiment_type"],
                          linking_tester_class=globals()[config["linking_tester_class"]],
                          the_results_directory=str(Path(directory).parent),
                          naming_style=config["naming_style"]
                          )

    @staticmethod
    def from_name_version_type(name, version, experiment_type, ):
        dummy_experiment = Experiment(name, version, experiment_type, None)
        return Experiment.from_directory(dummy_experiment.directory_full_path())

    def name_version_string(self):
        return self.name + ("_" + self.version if self.version else "")

    def class_name_version_string(self):
        return self.experiment_type + "_" + self.name_version_string()

    def directory_name(self):
        return self.class_name_version_string()

    def directory_full_path(self) -> str:
        return os.path.join(self.the_results_directory, self.directory_name())

    def experiment_name_for_tester(self):
        return self.name_version_string()

    def experiment_name_for_tester_legacy(self):
        return self.name

    def get_experiment_directory(self, create_if_not_exist=True):
        directory_full_path_status = self.what_is_this_path(self.directory_full_path())
        if directory_full_path_status == "file" or "test_dir" in directory_full_path_status:
            raise FileExistsError(
                f"{self.directory_full_path()}, which is the designated path for the experiment, already exists and is of type {directory_full_path_status}.")
        if directory_full_path_status == "experiment_dir":
            return self.directory_full_path()
        if create_if_not_exist:
            os.makedirs(self.directory_full_path(), exist_ok=True)
            return self.directory_full_path()
        return None

    def what_is_this_path(self, path):
        if not os.path.exists(path):
            return "nonexistent"
        if not os.path.isdir(path):
            return "file"
        if os.path.exists(os.path.join(path, "test_info.json")):
            with open(os.path.join(path, "test_info.json"), "r") as f:
                idx_info = json.load(f)
                if idx_info["experiment_name"] in self.experiment_name_for_tester():
                    return "related_test_dir"
                if idx_info["experiment_name"] in self.experiment_name_for_tester_legacy():
                    return "related_test_dir_legacy"
                return "unrelated_test_dir"
        else:
            if os.path.exists(os.path.join(path, "experiment_info.json")):
                return "experiment_dir"
            else:
                if len(os.listdir(path)) == 0:
                    return "empty_dir"

                return "test_dir_legacy"

    def is_test_directory(self, path):
        debug(f"Path {path} is {self.what_is_this_path(path)}")
        return self.what_is_this_path(path) in ["related_test_dir", "related_test_dir_legacy", "test_dir_legacy"]

    def is_experiment_directory(self, path):
        return self.what_is_this_path(path) == "experiment_dir"

    def register_test_directory(self, directory):
        debug(f"Trying to register discovered test directory {directory} ...")
        specs_dict = None
        if os.path.exists(os.path.join(directory, "test_info.json")):
            with open(os.path.join(directory, "test_info.json"), "r") as f:
                idx_info = json.load(f)
                specs_dict = idx_info.get("specs")
                if not specs_dict:
                    debug(f"  {directory}/test_info.json found but there is not specs.")
        else:
            debug(f"  {directory}/test_info.json does not exist.")

        if specs_dict is None:
            specs_dict = self.linking_tester_class.parse_conventional_name(directory, self.experiment_name_for_tester())
        if specs_dict is None:
            specs_dict = self.linking_tester_class.parse_conventional_name(directory,
                                                                           self.experiment_name_for_tester_legacy())
        if specs_dict is None:
            debug(f"  Cannot get specs from {directory}. Ignoring this directory.")
            return False

        # get the FULL specs dict
        default_specs_dict = self.linking_tester_class.default_specs()
        default_specs_dict.update(specs_dict)
        specs_dict = default_specs_dict
        print("SPECS_DICT____DEFAULT:", specs_dict)
        debug(f"  specs extracted: {TestSpecs.from_pure_dict(specs_dict).to_string()}")
        self.test_directories[TestSpecs.from_pure_dict(specs_dict).to_string()] = {"path": directory,
                                                                                   "status": "not open"}
        debug(f"  Registered. ")
        return TestSpecs.from_pure_dict(specs_dict).to_string()

    def find_existing_tests(self, force_update=True):
        if not force_update and len(self.test_directories) > 0:
            return self.test_directories
        debug("Find results directory ...")
        old_directories = self.test_directories
        self.test_directories = {}
        # find in The Results Directory
        if self.naming_style == "legacy" or self.naming_style is None:
            template_legacy = self.linking_tester_class.conventional_naming_template(
                self.experiment_name_for_tester_legacy())
            template = self.linking_tester_class.conventional_naming_template(self.experiment_name_for_tester())
            full_template_legacy = os.path.join(self.the_results_directory, template_legacy)
            full_template = os.path.join(self.the_results_directory, template)
            debug(
                f"Looking for test directories that follow {full_template}\n {glob.glob(full_template)} \n or {full_template_legacy} {glob.glob(full_template_legacy)}. ")
            full_list = glob.glob(full_template) + glob.glob(full_template_legacy)
            full_list = set(full_list)
            for directory in full_list:
                if self.is_test_directory(directory):
                    specs = self.register_test_directory(directory)
                    if specs in old_directories:
                        self.test_directories[specs] = old_directories[specs]

        # find in the experiment directory
        if self.naming_style == "manager" or self.naming_style is None:
            experiment_directory = self.get_experiment_directory(False)
            if experiment_directory:
                for directory in os.listdir(experiment_directory):
                    directory_full_path = os.path.join(experiment_directory, directory)
                    if self.is_test_directory(directory_full_path):
                        specs = self.register_test_directory(directory_full_path)
                        if specs in old_directories:
                            self.test_directories[specs] = old_directories[specs]
        return self.test_directories

    def set_specs_for_type_config(self, specs: ExperimentSpecs):
        self.type_config["specs"] = specs.to_dict()

    def set_specs_for_config(self, specs: ExperimentSpecs):
        self.get_experiment_directory()  # ensure that there is an experiment directory
        self.config["specs"] = specs.to_dict()

    def set_running_specs_for_config(self, specs: TestSpecs):
        self.get_experiment_directory()  # ensure that there is an experiment directory
        self.config["running_specs"] = specs.to_dict()

    def get_experiment_specs(self):
        default_specs = self.tester_default_specs
        type_defaults = self.type_config.get("specs", {})
        overrides = self.config.get("specs", {})
        specs = default_specs.copy()
        specs.update(type_defaults)
        specs.update(overrides)
        return ExperimentSpecs.from_pure_dict(specs)

    def get_all_test_specs_setup(self):
        return self.get_experiment_specs().expand()

    def get_running_specs(self):
        return self.config.get("running_specs", {})

    def create_tester_with_test_specs(self, test_specs: TestSpecs, experiment_name=None,
                                      experiment_version=None, savefile_root=None,
                                      experiment_specs_str=None):
        default_specs = self.linking_tester_class.default_specs()
        default_specs.update(test_specs.to_dict())
        test_specs = TestSpecs.from_pure_dict(default_specs)
        tester = self.linking_tester_class(experiment_name=experiment_name or self.experiment_name_for_tester(),
                                           experiment_version=experiment_version or self.version,
                                           savefile_root=savefile_root or self.directory_full_path(),
                                           experiment_specs_str=experiment_specs_str,
                                           **test_specs.to_dict())
        self.opened_tests[tester] = {"status": "open"}
        save_path = tester.find_save_path()
        self.test_directories[test_specs.to_string()] = {"path": save_path, "tester": tester, "status": "open"}
        return tester

    def run_tester(self, tester, callback_function=None, stop_event=None, **dynamic_running_specs):
        assert tester in self.opened_tests, "Not an opened tester!"
        if tester.is_running():
            tester.stop()
        print("TESTER", dynamic_running_specs)
        dynamic_model_info_list = dynamic_running_specs["dynamic_model_info_list"]

        # check the specs first
        test_specs = TestSpecs.from_pure_dict(dynamic_running_specs)
        # get defaults
        defaults = TestSpecs.from_pure_dict(self.config.get("running_specs", {}))
        print("DEFAULT", defaults)

        running_specs = defaults.realize(test_specs)
        print("RUNNINGSPECS", running_specs)
        if running_specs is None:
            raise ValueError(dynamic_running_specs)
        self.opened_tests[tester]["stop_event"] = stop_event
        self.opened_tests[tester]["callback_function"] = callback_function
        print("____", running_specs.to_dict())
        tester.standard_run_test_api(callback_function=callback_function, stop_event=stop_event,
                                     **running_specs.to_dict())

    def existing_directory_for_specs(self, specs: TestSpecs, keys=None):
        specs_str = specs.to_string()
        print()
        print(f"Existing directory for: {specs_str}")
        for key in self.test_directories.keys():
            print(f"    - {key}")
            if specs_str == key:
                print("        (HIT)")
        if specs_str in self.test_directories:
            if keys:
                return {key: self.test_directories[specs_str][key] for key in keys}
            return self.test_directories[specs_str]

        return None

    def run_all_tests(self, stop_event=None, **dynamic_running_specs):
        self.inscribe()
        self.find_existing_tests()
        all_specs = self.get_all_test_specs_setup()
        if len(all_specs) == 0:
            info("Nothing to do. Finished.")
            return
        augmented_headers = all_specs[0].specs_keys.copy()
        augmented_headers.extend(["status"])
        augmented_all_specs = [s.to_dict() for s in all_specs]

        for i, a in enumerate(augmented_all_specs):
            things = self.existing_directory_for_specs(all_specs[i])
            if things:
                status = things["status"]
            else:
                status = f"\033[90mnonexistent\033[0m"
            a.update(status=status)

        table, mws = "", None
        info()
        info("Begin experiment.")
        info()
        info()

        def refresh_table(move_cursor=False):
            nonlocal table, mws
            if gsm8k_rope.output_level != "debug" and move_cursor and self.verbosity:
                info(refresh_string(table), end="")
            table, mws = printable_table(augmented_all_specs, augmented_headers, mws)
            if self.verbosity:
                info(table)

        refresh_table()
        yield augmented_all_specs

        for i, specs in enumerate(all_specs):
            debug(f"Searching existing directory for specs: {specs.to_string()}")
            things = self.existing_directory_for_specs(specs)

            if things:

                directory = things["path"]
                debug(f"  Directory: {directory}")
                debug(f"  Things: {things}")
                test_name = str(Path(directory).name)
                directory = str(Path(directory).parent)

                # print(directory)
                # print(test_name)
                # input()
            else:
                directory = None
                test_name = None

            old_things_hook_handle = None

            if not things or "tester" not in things:
                print(f"Creating tester with specs {specs.to_string()}.")
                print("all existing:", self.test_directories)
                augmented_all_specs[i].update(status="\033[93mInitializing...\033[0m")
                refresh_table(True)
                yield augmented_all_specs
                tester = self.create_tester_with_test_specs(specs, savefile_root=directory,
                                                            experiment_specs_str=test_name)
                things = self.existing_directory_for_specs(specs)
                print("Now:", things)
                print(">all:", self.test_directories)

            else:
                debug("There is already a tester.")

                tester = things["tester"]
                old_things_hook_handle = things.get("hook_handle")
                print("Hook:", old_things_hook_handle)
                # if the tester already has a handle
                if old_things_hook_handle is not None:
                    if old_things_hook_handle.stop_event.is_set() or not tester.is_running():
                        tester.stop()  # wait until it stops
                        old_things_hook_handle = None
                        # now everything is clear, restart like nothing happens

            if old_things_hook_handle:
                if stop_event:
                    old_things_hook_handle.set_stop_event(stop_event)
                q = old_things_hook_handle.auxiliary["queue"]
                old_things_hook_handle.hook()
            else:
                # create a new handle
                things["hook_handle"] = Hook()  # register hook handle
                q = mp.Queue()
                things["hook_handle"].auxiliary = {"queue": q}
                if stop_event:
                    things["hook_handle"].set_stop_event(stop_event)
            hook_handle = things["hook_handle"]

            def callback_here(callback_results):
                total = callback_results["total_tasks"]
                finished = callback_results["finished_tasks"]

                def get_color(force_truecolor=False):
                    starting_color = (255, 0, 0)
                    ending_color = (0, 230, 0)
                    ratio = finished / total
                    red = int(starting_color[0] * (1 - ratio) + ending_color[0] * (ratio))
                    green = int(starting_color[1] * (1 - ratio) + ending_color[1] * (ratio))
                    rgb_color = f"\033[38;2;{red};{green};0m"
                    cube_r = int(red / 42.667)
                    cube_g = int(green / 42.667)
                    cube_b = 0
                    cube_color = f"\033[38;5;{16 + cube_r * 36 + cube_g * 6 + cube_b}m"
                    colorterm = os.environ.get('COLORTERM', '')
                    if 'truecolor' in colorterm or '24bit' in colorterm or force_truecolor:
                        return rgb_color
                    return cube_color

                things["status"] = f"{get_color(True)}Running ({finished}/{total})\033[0m"

                augmented_all_specs[i].update(status=f"{get_color()}Running ({finished}/{total})\033[0m")
                refresh_table(True)
                augmented_all_specs[i].update(status=f"{get_color(True)}Running ({finished}/{total})\033[0m")

                q.put(augmented_all_specs)

            debug(f"Running tester with dynamic specs: {dynamic_running_specs}")

            # start a new thread and run the tester
            def worker(_tester, callback_function, _stop_event, _dynamic_running_specs):
                self.run_tester(_tester, callback_function=callback_function, stop_event=_stop_event, **_dynamic_running_specs)
                # self.run_tester(tester, callback_function=callback_here, stop_event=hook_handle.stop_event,
                #                 **dynamic_running_specs)
                q.put(None)

            if old_things_hook_handle:
                # there is a thread already running; do not start new thread
                process = None
            else:
                process = mp.Process(target=worker, args=(tester, callback_here,
                                                          hook_handle.stop_event, dynamic_running_specs))
                process.start()
            try:
                while True:
                    callback_result = q.get()
                    if callback_result is None:  # worker finishes
                        break
                    yield callback_result  # yielded: augmented_all_specs


            except Exception as e:
                info(f"Detected exception of type {type(e)}: {e}. Stopping the worker before handling exception.")
                hook_handle.stop()
                if process:
                    process.join()
                raise e
            print("End")
            things["status"] = "\033[32mFinished\033[0m"
            augmented_all_specs[i].update(status="\033[32mFinished\033[0m")
            refresh_table(True)
            yield augmented_all_specs


class FillerExperimentPlanExample:
    def __init__(self, name, version):
        self.experiment = Experiment(name, version,
                                     "filler",
                                     FillerTester)
        self.bs = BackendSubmitter()

    def define_experiment(self):
        self.experiment.set_specs_for_config(ExperimentSpecs.from_pure_dict({
            "filler_type": ["sequence_space", "essay", "fake_distance"],
            "retrieval_format": [False, True],
            "problem_set": ["simple"]
        }))
        self.experiment.set_running_specs_for_config(TestSpecs.from_pure_dict({
            "context_lengths_min": 0,
            "context_lengths_max": 30000,
            "context_lengths_num_intervals": 9,
            "url_list": {"type": "dynamic"},
            "skip_existing": True,
        }))

    def get_nodes(self, **kwargs):
        self.bs.submit(verbal=True,
                       **kwargs
                       )

    def run_all_experiments(self):
        for _ in self.experiment.run_all_tests(url_list=",".join(self.bs.node_list_with_ports), ):
            continue


class ExperimentManager:
    default_the_results_directory = "results/rope"

    def __init__(self, the_results_directory=None, verbosity=False):
        self.kwargs_for_running_experiment = None
        self.list_of_experiments = {}
        self.bs = None
        self.username = os.environ.get('USER', os.environ.get('USERNAME'))
        self.global_config = Config(".global")
        self.bs_config = Config(".bs")
        self.user_config = Config(".experiment_manager_config", os.path.expanduser("~"))
        self.the_results_directory = the_results_directory or self.global_config.get("the_results_directory")
        self.resource_lock = threading.Lock()
        self.auto_job_thread = None
        self.verbosity = verbosity

    def get_username(self):
        return self.username or "unknown"

    def get_user_info(self):
        return {
            "username": self.get_username(),
            "auto_submit": self.user_config.get("auto_submit", {"mode": "off"})
        }

    def get_bs(self) -> BackendSubmitter:
        if self.bs is None:
            with self.resource_lock:
                launching_jobs = self.bs_config.get("launching_jobs", None)
                if launching_jobs:
                    launching_jobs = {node: set(devices) for node, devices in launching_jobs.items()}
            self.bs = BackendSubmitter(old_launching_jobs=launching_jobs)
        return self.bs

    def get_slurm_jobs(self):
        squeue_results = self.get_bs().get_slurm_jobs().filter(USER=self.get_username())
        squeue_results = squeue_results.get_columns(["JOBID", "NODES", "ST", "TIME", "NODELIST(REASON)"])
        nodes_cnt = 0
        for nodes in squeue_results["NODES"]:
            nodes_cnt += int(nodes)
        with self.resource_lock:
            return {"headers": squeue_results.headers, "data": squeue_results.data, "nodes_cnt": nodes_cnt,
                    "auto_submit": self.user_config.get("auto_submit", {"mode": "off"})}

    def get_slurm_settings(self):
        return self.user_config.get("slurm_default_kwargs", "No settings. You have to provide one!\n"
                                                            "Template: provide at least these items.\n"
                                                            '{\n'
                                                            '    "account": ACCOUNT_NAME,\n'
                                                            '    "partition": PARTITION,\n'
                                                            '    "nodes": NUMBER_OF_NODES,\n'
                                                            '    "time": HH:MM:SS (BE STRICT ON THIS!)\n'
                                                            '}')

    def manage_auto_submit_job(self, mode="on", nodes="8"):
        with self.resource_lock:
            self.user_config["auto_submit"] = {"mode": mode, "nodes": nodes}

        def auto_submit_job_worker():
            try:
                while True:
                    with self.resource_lock:
                        auto_submit_settings = self.user_config.get("auto_submit", {"mode": "off", "nodes": "0"})
                        if auto_submit_settings["mode"] == "off":
                            self.auto_job_thread = None
                            return
                    nodes = auto_submit_settings["nodes"]
                    nodes = int(nodes)
                    jobs = self.get_slurm_jobs()
                    if jobs["nodes_cnt"] < nodes:
                        self.submit_slurm_job()
                    time.sleep(2)
            except Exception:
                with self.resource_lock:
                    auto_submit_settings = self.user_config.get("auto_submit", {"mode": "off", "nodes": "0"})
                    self.user_config["auto_submit"] = {"mode": "off", "nodes": auto_submit_settings.get("nodes", "0")}
                return

        if mode == "on" and not self.auto_job_thread:
            self.auto_job_thread = threading.Thread(target=auto_submit_job_worker, daemon=True)
            self.auto_job_thread.start()

    def cancel_slurm_job(self, jobid):
        return self.get_bs().cancel_slurm_job(jobid)

    def import_slurm_settings_from_file(self, filename):
        kwargs = parse_sbatch_file(filename)
        self.user_config["slurm_default_kwargs"] = kwargs
        return kwargs

    def submit_slurm_job(self, **kwargs):
        sbatch_default_kwargs = self.user_config.get("slurm_default_kwargs", {})
        # print("submit: default kwargs", sbatch_default_kwargs)
        sbatch_default_kwargs.update(kwargs)
        self.user_config["slurm_default_kwargs"] = sbatch_default_kwargs
        return self.get_bs().submit_slurm_job(**sbatch_default_kwargs)

    def get_nodelist_with_status(self):
        self.get_bs().update_nodelist()
        launching_jobs, _, _ = self.get_bs().get_launching_jobs_nodelist_node_port_status_copy_thread_safe()
        launching_jobs = {node: list(devices) for node, devices in launching_jobs.items()}
        with self.resource_lock:
            self.bs_config["launching_jobs"] = launching_jobs
            # print("SV1", launching_jobs)
        return self.get_bs().get_nodelist_with_status()

    def get_history(self):
        return self.user_config.get("history", {"submit_backend_jobs_history": []})

    def submit_backend_jobs(self, verbal=True, **kwargs):
        # register kwargs
        history = self.user_config.get("history", {"submit_backend_jobs_history": []})
        submit_backend_history = history.get("submit_backend_jobs_history", [])
        submit_backend_history.append(kwargs)
        history["submit_backend_jobs_history"] = submit_backend_history
        self.user_config["history"] = history

        # gsm8k_rope.debug("Submit b")
        bs = self.get_bs()
        # gsm8k_rope.debug("Submit")
        for whatever in bs.submit(verbal=verbal, interactive=True, **kwargs):
            launching_jobs, _, _ = self.get_bs().get_launching_jobs_nodelist_node_port_status_copy_thread_safe()
            launching_jobs = {node: list(devices) for node, devices in launching_jobs.items()}
            with self.resource_lock:
                self.bs_config["launching_jobs"] = launching_jobs
            # print("SV", launching_jobs)
            yield whatever

    def kill_backend_jobs(self):
        self.get_bs().kill_all()

    def set_the_results_library(self, directory=None, save_to_config=True):
        if directory is None:
            directory = self.default_the_results_directory
        if save_to_config:
            self.global_config["the_results_directory"] = directory

    def lazy_initialization(self):
        self.list_of_experiments = {}
        self.set_the_results_library()
        self.get_bs()

    def append_experiment_and_set_active(self, experiment):
        self.list_of_experiments[experiment.directory_full_path()] = experiment
        experiment.find_existing_tests()

    def define_experiment(self, experiment_name,
                          experiment_version,
                          experiment_type,
                          linking_tester_class,
                          experiment_specs_dict,
                          default_test_specs_dict,
                          open_experiment=False,
                          ok_if_exists=True,
                          ):
        # create an experiment
        # check if experiment is open
        experiment = Experiment(experiment_name, experiment_version, experiment_type, linking_tester_class,
                                self.the_results_directory, )
        if not ok_if_exists and experiment.exists():
            raise FileExistsError(f"Experiment at {experiment.get_experiment_directory()} already exists")
        experiment.inscribe()
        experiment.set_specs_for_config(ExperimentSpecs.from_pure_dict(experiment_specs_dict))
        experiment.set_running_specs_for_config(TestSpecs.from_pure_dict(default_test_specs_dict))

        if open_experiment:
            self.append_experiment_and_set_active(experiment)

        return experiment

    def define_experiment_type(self, experiment_type, experiment_type_specs_dict, ok_if_exists=True):

        global_config_experiment_types = self.global_config.get("experiment_types", [])
        global_config_experiment_types_set = set(global_config_experiment_types)
        if not ok_if_exists and experiment_type in global_config_experiment_types_set:
            raise FileExistsError(f"Experiment type {experiment_type} already exists")
        global_config_experiment_types_set.update(experiment_type)
        self.global_config["experiment_types"] = list(global_config_experiment_types_set)
        experiment_type_config = Config(experiment_type)
        experiment_type_config["specs"] = experiment_type_specs_dict

    def open_experiment(self, experiment_name=None, experiment_version=None, experiment_type=None,
                        exp_dir=None, ignore_if_open=True):
        if exp_dir is not None:
            temp_experiment = Experiment.from_directory(exp_dir)
        else:
            assert experiment_name is not None and experiment_version is not None and experiment_type is not None
            temp_experiment = Experiment.from_name_version_type(experiment_name, experiment_version, experiment_type)
        if temp_experiment:
            if temp_experiment.directory_full_path() in self.list_of_experiments and ignore_if_open:
                print(
                    f"Experiment already open: {self.list_of_experiments[temp_experiment.directory_full_path()].test_directories}")
                return self.list_of_experiments[temp_experiment.directory_full_path()]

            print("Existing paths", list(self.list_of_experiments.keys()))
            print("This path", temp_experiment.directory_full_path())

            self.append_experiment_and_set_active(temp_experiment)
            temp_experiment.verbosity = self.verbosity
        return temp_experiment

    def find_and_open_all_experiments_in_dir(self):
        for directory in os.listdir(self.the_results_directory):
            self.open_experiment(exp_dir=os.path.join(self.the_results_directory, directory))
        return list(self.list_of_experiments.values())

    def update_nodelist(self):
        self.get_bs().update_nodelist()
        return self.bs.node_list_with_ports

    def set_kwargs_for_running_experiment(self, **kwargs):
        self.kwargs_for_running_experiment = kwargs

    def update_api_key(self, api_name, api_key):
        print("!!!!!!!!!!!!!!!UPDATE")
        if self.user_config.get("api_keys") is None:
            self.user_config["api_keys"] = {}
        self.user_config["api_keys"][api_name] = api_key
        self.user_config.sync()

    def get_api_keys(self):
        return self.user_config.get("api_keys", {})

    def run_one_experiment(self, experiment, interactive=False, update_node_list=True,
                           stop_event=None, **kwargs):
        if self.kwargs_for_running_experiment and len(kwargs) == 0:
            kwargs = self.kwargs_for_running_experiment

        if update_node_list:
            self.update_nodelist()
        if kwargs.get("dynamic_model_info_list"):
            dynamic_model_info_list = kwargs.get("dynamic_model_info_list")
        else:
            if experiment.resource_type() == "gpu":
                if kwargs.get("dynamic_model_info_list_for_gpu"):
                    dynamic_model_info_list = kwargs.get("dynamic_model_info_list_for_gpu")
                else:
                    dynamic_model_info_list = [self.bs.node_list_with_ports]
            else:
                keyname = experiment.resource_type().replace("_api_key", "")
                if kwargs.get(f"dynamic_model_info_list_for_{keyname}"):
                    dynamic_model_info_list = kwargs.get(f"dynamic_model_info_list_for_{keyname}")
                else:
                    api_key = self.user_config.get(keyname)
                    if api_key is None:
                        raise ValueError(f"{keyname} is not provided.")
                    dynamic_model_info_list = [[api_key for _ in range(1)]]  # just one instance
        keys_of_kwargs = list(kwargs.keys())
        for key in keys_of_kwargs:
            if key.startswith("dynamic_model_info_list"):
                kwargs.pop(key)
        for interactive_result in experiment.run_all_tests(stop_event=stop_event,
                                                           dynamic_model_info_list=dynamic_model_info_list, **kwargs):
            if interactive:
                yield {
                    "experiment": experiment.class_name_version_string(),
                    "intermediate": True,
                    "status": "running",
                    "augmented_all_specs": interactive_result
                }
        if interactive:
            yield {
                "experiment": experiment.class_name_version_string(),
                "intermediate": False,
                "status": "experiment finished"
            }

    def run_all_experiments(self, interactive=False, update_node_list=True, **kwargs):
        for experiment in self.list_of_experiments:
            for whatever in self.run_one_experiment(experiment, interactive, update_node_list, **kwargs):
                yield whatever
        yield {"intermediate": False, "status": "all finished"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kill', action="store_true", help="Kill all Megatron servers.")
    parser.add_argument('--model_path', )
    parser.add_argument('--tokenizer_path', )
    parser.add_argument('--rope_base', default="500000")
    parser.add_argument("--model_type", default="others", choices=["llama3", "others"])
    parser.add_argument("--attn_impl", default="te")
    parser.add_argument("--name", default="llama3.1_instruct")
    parser.add_argument("--version", default="v0327")
    parser.add_argument("--submit_backend_only", action="store_true")
    args = parser.parse_args()
    # gsm8k_rope.output_level = "debug"
    plan = FillerExperimentPlanExample(args.name, args.version)
    if args.kill:
        plan.bs.kill_all()
        exit(0)

    plan.get_nodes(model_path=args.model_path, tokenizer_path=args.tokenizer_path,
                   rope_base=args.rope_base, model_type=args.model_type,
                   attn_impl=args.attn_impl)
    if args.submit_backend_only:
        exit(0)
    plan.define_experiment()

    plan.run_all_experiments()
