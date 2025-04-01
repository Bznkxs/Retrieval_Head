import os
import subprocess
import sys
import tempfile

from managing.terminal_formatting import parse_nodes


def run_squeue_and_return(*args: str):
    completed_process = subprocess.run(["squeue"] + list(args), capture_output=True)
    return completed_process.stdout.decode("utf-8")

class DataMatrix:
    def __init__(self, data, headers=None):
        self.data = data  # list of lists
        self.headers = headers

    def get_column_index(self, column_name):
        if self.headers is None:
            raise ValueError("headers must be provided when using string index")
        if column_name not in self.headers:
            raise ValueError(f"Name {column_name} not in headers {self.headers}")
        return self.headers.index(column_name)

    def __getitem__(self, index):
        if isinstance(index, str):
            column_index = self.get_column_index(index)
            return [row[column_index] for row in self.data]
        if isinstance(index, int) or isinstance(index, slice):
            return self.data[index]
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Only 2D indexing is supported")
            index_0, index_1 = index
            assert ((isinstance(index_0, int) or isinstance(index_0, slice)) and
                    (isinstance(index_1, int) or isinstance(index_1, slice))), \
                "Only integer or slice indexing is supported"

            return [row[index_1] for row in self.data[index_0]]

        raise ValueError("Only string, integer, slice, and 2-tuple indexing is supported")

    def get_columns(self, column_names):
        data = []
        target_columns = []
        headers = []
        for key in column_names:
            if key in self.headers:
                target_columns.append(self.headers.index(key))
                headers.append(key)
        for row in self.data:
            data.append([])
            for col in target_columns:
                data[-1].append(row[col])
        return DataMatrix(data=data, headers=headers)


    def filter(self, **kwargs):
        # return a SqueueResults object
        data = []
        target_columns = {}
        # get target columns of keys in kwargs
        for key, value in kwargs.items():

            if key in self.headers:
                target_columns[self.headers.index(key)] = value

        def match(item, condition):
            if isinstance(condition, str):
                return item == condition
            else:
                return item in condition

        for entry in self.data:
            in_flag = True
            for column in target_columns:
                if not match(entry[column], target_columns[column]):
                    in_flag = False
                    break

            if in_flag:
                data.append(entry)

        return DataMatrix(data=data, headers=self.headers)

    def show(self):
        for entry in self.data:
            for idx in range(len(entry)):
                print(f"{self.headers[idx]}: {entry[idx]}", end=', ')
            print()

class SqueueResults(DataMatrix):
    possible_kwargs = ['ACCOUNT', 'TRES_PER_NODE', 'MIN_CPUS', 'MIN_TMP_DISK', 'END_TIME', 'FEATURES', 'GROUP', 'OVER_SUBSCRIBE', 'JOBID', 'NAME', 'COMMENT', 'TIME_LIMIT', 'MIN_MEMORY', 'REQ_NODES', 'COMMAND', 'PRIORITY', 'QOS', 'REASON', 'ST', 'USER', 'RESERVATION', 'WCKEY', 'EXC_NODES', 'NICE', 'S:C:T', 'JOBID', 'EXEC_HOST', 'CPUS', 'NODES', 'DEPENDENCY', 'ARRAY_JOB_ID', 'GROUP', 'SOCKETS_PER_NODE', 'CORES_PER_SOCKET', 'THREADS_PER_CORE', 'ARRAY_TASK_ID', 'TIME_LEFT', 'TIME', 'NODELIST', 'CONTIGUOUS', 'PARTITION', 'PRIORITY', 'NODELIST(REASON)', 'START_TIME', 'STATE', 'UID', 'SUBMIT_TIME', 'LICENSES', 'CORE_SPEC', 'SCHEDNODES', 'WORK_DIR']
    def __init__(self, string_results=None, headers=None, data=None):
        if string_results:
            string_results = string_results.strip()
            self.string_results = string_results
            self.lines = string_results.split("\n")
            headers = self.lines[0].split("|")
            data = [line.split("|") for line in self.lines[1:]]
        super().__init__(data, headers)

    def filter(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        for key in kwargs:
            if key not in self.possible_kwargs:
                raise ValueError(f"Name {key} not in {self.possible_kwargs}")
        dm = super().filter(**kwargs)
        return SqueueResults(headers=dm.headers, data=dm.data)



    def get_nodes(self, *keywords_to_retrieve):
        nodes = self["NODELIST(REASON)"]
        node_dict = {}
        columns_of_interest = {k: self[k] for k in keywords_to_retrieve}
        for idx, node_string in enumerate(nodes):
            new_node_list = parse_nodes(node_string)
            for node in new_node_list:
                node_dict[node] = {k: columns_of_interest[k][idx]
                                   for k in keywords_to_retrieve}
        return node_dict

def squeue_for_parsing():
    string_results = run_squeue_and_return("--format", "%all")
    return SqueueResults(string_results)

def submit_slurm_job(account, partition, nodes, time, gres="gpu:4", **kwargs):
    """
    time must be of hh:mm:ss
    """
    sbatch_str = "#!/bin/bash\n"
    submit_kwargs = dict(account=account, partition=partition, nodes=nodes, time=time, gres=gres)
    submit_kwargs.update(job_name="slurm_job", mem="0", ntasks_per_node="1", cpus_per_task="288")
    submit_kwargs.update(kwargs)
    for kw in submit_kwargs:
        kw_in_str = kw.replace("_", "-")
        sbatch_str += f"#SBATCH --{kw_in_str}={submit_kwargs[kw]}\n"
    sbatch_str += "srun hostname > hostfile\nsleep "
    hh, mm, ss = time.split(":")
    sbatch_str += f"{hh}h {mm}m {ss}s\n"

    with tempfile.TemporaryDirectory() as tmpdirname:  # https://stackoverflow.com/questions/3223604/how-do-i-create-a-temporary-directory-in-python
        batch_file = os.path.join(tmpdirname, "run_slurm_job.sh")
        with open(batch_file, "w") as f:
            f.write(sbatch_str)
        completed_process = subprocess.run(["sbatch", batch_file])
        return completed_process.returncode

def cancel_slurm_job(jobid):
    completed_process = subprocess.run(["scancel", str(jobid)])
    return completed_process.returncode


def parse_sbatch_file(filename):
    with open(filename) as f:
        file = f.read()
        lines = file.split("\n")
        kwargs = {}
        for line in lines:
            line = line.strip()
            if line.startswith("#SBATCH"):
                line = line.replace("#SBATCH --", "")
                line = line.split("#")[0]  # remove comments
                key, value = line.split("=", maxsplit=1)
                kwargs[key] = value
        return kwargs


if __name__ == '__main__':
    squeue_results = squeue_for_parsing().filter(STATE="RUNNING", USER=sys.argv[1] if len(sys.argv) > 1 else None)
    for k, v in squeue_results.get_nodes("JOBID", "USER").items():
        print(f"{k}: {v}")
    print(list(squeue_results.get_nodes().keys()))
