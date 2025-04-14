import copy
import json
import os
import subprocess
import sys
import threading
import time
import tempfile

import requests

import gsm8k_rope

from managing.slurm_related import submit_slurm_job, cancel_slurm_job, squeue_for_parsing
from managing.ssh_and_backend import launch_backend, brute_force_kill_pt_main_thread
from managing.terminal_formatting import cartesian_product_of_nodes_and_ports, get_ansi_foreground_color, \
    printable_table, refresh_string


def check_megatron_server_status(node_name, port_range=None):
    if port_range is None:
        port_range = range(5000, 5004)
    # 4 possible nodes
    responses = []
    for port in port_range:
        url = f"http://{node_name}:{port}/api/info"
        try:
            response = requests.put(url, timeout=.5)
            if response.status_code == 200:
                responses.append({"info": json.loads(response.text), "server": True,
                                  "running": True, "main_port": port, "status_code": 200})
            else:
                responses.append({"server": True, "running": True, "main_port": port,
                                  "status_code": response.status_code})
        except requests.exceptions.ReadTimeout as e:
            responses.append({"server": True, "running": False, "main_port": port, "Exception": "timeout"})
        except Exception as e:
            responses.append({"server": False, "running": False, "main_port": None, "Exception": str(e)})
    for i in range(len(responses)):
        if responses[i]["server"] and responses[i]["running"]:
            try:
                world_size = responses[i]["info"]["world_size"]
                for j in range(world_size - 1):
                    responses[i + j] = {"server": False, "running": True, "main_port": responses[i]["main_port"],
                                        "info": responses[i]["info"]}
            except Exception:
                print("NO WORLD SIZE")

    return responses


all_possible_stati_and_display_colors = {
    "running": "green",
    "not head node": "green",
    "launching": "yellow",
    "no job": "gray",
    "unknown": "red",
    "timeout": "red"
}

def explain_server_status(server_status):
    """
    Running Server: server == True, running == True
    * Get info successful: status_code == 200; info (world_size, in_use)
    * Get info failed: status_code != 200
    Not Head Node: server == False, running == True
    * only when head node get info successful
    Timeout: server == True, running == False, Exception == "timeout"
    Not Running: server == False, running == False
    Others: (top priority)
    * Launching: others == "launching"
    """
    if server_status.get("others"):
        return server_status["others"]
    if server_status["server"] and server_status["running"]:
        return "running"
    if not server_status["server"] and server_status["running"]:
        return "not head node"
    if server_status["server"] and not server_status["running"] and server_status["Exception"] == "timeout":
        return "timeout"
    if not server_status["server"] and not server_status["running"]:
        return "no job"
    return "unknown"



def get_default_logfile_name(node_name, device):
    return os.path.abspath(os.path.join(os.curdir, "remote_logs", f"t{node_name}:{device}.log"))


def submit_one_job_on_node(node_name, device,
                           megatron_root="/u/yufengd4/wbbzy/Megatron-LM",
                           remote_script="bash apptainer_run_text_generation.sh",
                           model_path="/work/nvme/bdtq/yufengd4/llama_3.1_mg",
                           tokenizer_path="/work/nvme/bdtq/mtian8/models/HF_model/Llama-3.1-8B-Instruct",
                           rope_base="500000",
                           model_type="llama3",
                           attn_impl="te",
                           ):
    log_file = get_default_logfile_name(node_name, device)
    # print(log_file)
    launch_backend(node_name, megatron_root,
                   remote_script, log_file,
                   f'{model_path} {tokenizer_path} {rope_base} {model_type} {attn_impl} {device}'
                   )
    return log_file

def find_free_gpu_and_submit_one_job_on_node(node_name, excluded=None, **kwargs):
    """
    Warning: this is not thread-safe, as it returns before the server runs. Make sure
    to mark the GPU that this function runs as "launching".
    returns: device_id, log_file_path
    """
    server_status = check_megatron_server_status(node_name)
    for i in range(len(server_status)):
        if excluded is not None and i in excluded:
            continue
        if explain_server_status(server_status[i]) == "no job":  # risky
            # print("Found free gpu:",  node_name, i, "status", server_status)
            log_file = submit_one_job_on_node(node_name,
                                              device=i,
                                              **kwargs)
            return i, log_file
    return None, None


class BackendSubmitter:
    # todo: only one instance may exist
    def __init__(self, node_list=None, old_launching_jobs=None):
        """
        node_list: if None, use all running nodes.
        """
        self.node_list = []
        self.launching_jobs = None
        self.node_port_status = {}
        self.node_list_with_ports = []

        self.all_squeue_results = None

        self.status_lock = threading.Lock()
        self.submit_lock = threading.Lock()
        self.kill_lock = threading.Lock()
        self.slurm_lock = threading.Lock()
        self.submit_stop_event = threading.Event()
        self.submit_stop_event.set()
        self.update_nodelist(node_list, old_launching_jobs)


    def get_slurm_jobs(self):
        with self.slurm_lock:
            self.all_squeue_results = squeue_for_parsing()
            return self.all_squeue_results

    def submit_slurm_job(self, account, partition, nodes, time, gres="gpu:4", **kwargs):
        with self.slurm_lock:
            return submit_slurm_job(account, partition, nodes, time, gres, **kwargs)

    def cancel_slurm_job(self, jobid):
        with self.slurm_lock:
            return cancel_slurm_job(jobid)

    def set_launching_jobs_and_status_thread_safe(self, launching_jobs=None, node_port_status=None):
        with self.status_lock:
            if launching_jobs:
                self.launching_jobs = launching_jobs
            if node_port_status:
                self.node_port_status = node_port_status

    def get_launching_jobs_nodelist_node_port_status_copy_thread_safe(self):
        with self.status_lock:
            launching_jobs = {}
            for node in self.launching_jobs:
                launching_jobs[node] = set()
                for i in self.launching_jobs[node]:
                    launching_jobs[node].add(i)
            node_list = [i for i in self.node_list]
            node_port_status = copy.deepcopy(self.node_port_status)
            return launching_jobs, node_list, node_port_status


    def update_nodelist(self, node_list=None, old_launching_jobs=None):
        """
        Thread safe. Used lock: status_lock
        """
        if node_list is None:

            squeue_results = self.get_slurm_jobs().filter(STATE="RUNNING", USER=os.environ["USER"])
            node_list = list(squeue_results.get_nodes().keys())
        # print(self.status_lock)
        with self.status_lock:
            self.node_list = node_list
            if old_launching_jobs is None:
                old_launching_jobs = self.launching_jobs
            if old_launching_jobs is None:
                old_launching_jobs = {}
            self.launching_jobs = {node: old_launching_jobs.get(node, set()) for node in node_list}
            # print(self.launching_jobs)
            self.node_list_with_ports = cartesian_product_of_nodes_and_ports(self.node_list, [5000, 5001, 5002, 5003])
            old_node_port_status = self.node_port_status
            self.node_port_status = {}
            for node_port in self.node_list_with_ports:
                node, port = node_port.split(":")
                self.node_port_status[node_port] = check_megatron_server_status(node, [port])[0]
                # print(f"Check status of {node_port}: {explain_server_status(self.node_port_status[node_port])}")
                if int(port) - 5000 in self.launching_jobs[node]:
                    status = explain_server_status(self.node_port_status[node_port])
                    if status == "no job" or status == "launching":
                        self.node_port_status[node_port] = {"others": "launching"}
                    else:
                        self.launching_jobs[node].remove(int(port) - 5000)
                        print(f"port {port}: status={status}. removed from launching_jobs")

                if old_node_port_status.get(node_port):
                    self.node_port_status[node_port]["log_file"] = old_node_port_status[node_port].get("log_file")
            # print(">", self.launching_jobs)

    def get_nodelist_with_status(self):
        # print(self.status_lock)
        ret = self.get_launching_jobs_nodelist_node_port_status_copy_thread_safe()[2]
        for k in ret:
            ret[k]["readable"] = explain_server_status(ret[k])
            ret[k]["display_color"] = all_possible_stati_and_display_colors[ret[k]["readable"]]
        return ret

    def update_status_and_launching_jobs(self):
        with self.status_lock:
            for node in self.node_list:
                status = check_megatron_server_status(node)
                # update status
                for i in range(len(status)):
                    readable_status = explain_server_status(status[i])

                    self.node_port_status[f"{node}:500{i}"].update(status[i])
                    if readable_status == "running" and i in self.launching_jobs[node]:
                        self.launching_jobs[node].discard(i)
                        self.node_port_status[f"{node}:500{i}"].pop("others")


    def submit(self, verbal=False, interactive=False, **kwargs):
        gsm8k_rope.debug("Submit lock:", self.submit_lock)

        with self.submit_lock:
            self.submit_stop_event.clear()
            mw = None
            printable_str = ""
            gsm8k_rope.debug("?")
            while True:
                if self.submit_stop_event.is_set():
                    if verbal:
                        print("Stopped through signal")
                    return
                free_gpu_flag = False  # there exists a free gpu
                launching_jobs, node_list, node_port_status = self.get_launching_jobs_nodelist_node_port_status_copy_thread_safe()
                gsm8k_rope.debug(node_list)
                for node in node_list:
                    free_gpu, log_file = find_free_gpu_and_submit_one_job_on_node(node, launching_jobs[node], **kwargs)
                    if free_gpu is not None:
                        launching_jobs[node].add(free_gpu)
                        node_port_status[f"{node}:500{free_gpu}"] = {"others": "launching", "log_file": log_file}
                        free_gpu_flag = True
                        if verbal:
                            print(f"Submitted job to {node}:500{free_gpu}. Logs saved to {log_file}")
                            if mw:
                                mw = None
                                printable_str = None
                        if interactive:
                            yield {"action": "submit", "node": node, "port": f"500{free_gpu}", "log": log_file,
                                   "status": node_port_status }
                if self.submit_stop_event.is_set():
                    if verbal:
                        print("Stopped through signal")
                    return
                self.set_launching_jobs_and_status_thread_safe(launching_jobs, node_port_status)

                if free_gpu_flag is False:
                    self.update_status_and_launching_jobs()
                    launching_jobs, node_list, node_port_status = self.get_launching_jobs_nodelist_node_port_status_copy_thread_safe()
                    if verbal:
                        table = []
                        for node in node_list:
                            table_row = {}
                            for i in range(5000, 5004):
                                port = str(i)
                                status = node_port_status[f"{node}:{port}"]
                                readable_status = explain_server_status(status)
                                color_literal = all_possible_stati_and_display_colors[readable_status]
                                foreground_color_ansi_sequence = get_ansi_foreground_color(color_literal)
                                status_with_color = foreground_color_ansi_sequence + readable_status + "\033[0m"
                                table_row[port] = status_with_color
                            table.append(table_row)
                        if len(node_list):
                            if printable_str and gsm8k_rope.output_level != "debug":
                                print(refresh_string(printable_str))
                            printable_str, mw = printable_table(table, [str(i) for i in range(5000,5004)], mw)
                            print(printable_str)
                    if interactive:
                        yield {"action": "wait", "status": node_port_status}
                    with self.status_lock:
                        finished_flag = sum([len(self.launching_jobs[node]) for node in self.launching_jobs]) == 0
                        print("FLAG", self.launching_jobs)
                    if finished_flag:
                        if verbal:
                            print("Finished submission")
                            print()
                        if interactive:
                            yield {"action": "finished", "status": node_port_status}
                        break

                    time.sleep(.2)
    def kill_all(self):
        """
        Thread safe. Used: kill_lock, status_lock
        """
        with self.kill_lock:
            self.submit_stop_event.set()
            flg = False
            if self.submit_lock.locked():
                print("Wait for submit to finish")
                flg = True
            with self.submit_lock:
                if flg:
                    print("fin.")
                with self.status_lock:
                    nodelist = self.node_list.copy()
                for node in nodelist:
                    brute_force_kill_pt_main_thread(node)
                with self.status_lock:
                    self.node_list = []
                    self.launching_jobs = {}
                    self.node_port_status = {}
                    self.node_list_with_ports = []
                self.update_nodelist()


if __name__ == '__main__':
    squeue_results = squeue_for_parsing().filter(STATE="RUNNING", USER=sys.argv[1])
    node_list = list(squeue_results.get_nodes().keys())
    print(f"Node list for user {sys.argv[1]}: {node_list}")
    bs = BackendSubmitter(node_list)
    bs.submit(verbal=True)
