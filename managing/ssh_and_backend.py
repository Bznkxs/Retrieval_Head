import subprocess
from pathlib import Path
import os
def launch_backend(remote_host, project_root, remote_script, log_file, args):
    os.makedirs(str(Path(log_file).parent), exist_ok=True)
    # print("->", f"nohup bash -c 'cd {project_root} && {remote_script} {args}' > {log_file} 2>&1 &")
    subprocess.run([
            "ssh",
            remote_host,
            f"nohup bash -c 'cd {project_root} && {remote_script} {args}' > {log_file} 2>&1 &"
        ],
    )

def brute_force_kill_pt_main_thread(remote_host):
    subprocess.run([
        "ssh",
        remote_host,
        'ps -u $USER -o pid=,comm= | grep "pt_main_thread" | awk "{print \\$1}" | xargs -r kill -9'
    ])

if __name__ == "__main__":
    launch_backend("gh013", ".",
                   "echo 'hello world'", "t1.log",
                   '')