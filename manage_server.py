import inspect
import json
import os
import subprocess
import threading
import sys

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from manage import Experiment, ExperimentManager, Hook
import gsm8k_rope
from datetime import datetime
# gsm8k_rope.output_level = "debug"
app = Flask(__name__)
manager = ExperimentManager()
manager.lazy_initialization()



class NotProvidedKwarg:
    pass


def dynamic_kwargs(keyword_list, **provided_kwargs):
    gsm8k_rope.debug("In dynamic kwargs")
    kwargs = {}
    for keyword in keyword_list:
        if not isinstance(provided_kwargs.get(keyword, NotProvidedKwarg()), NotProvidedKwarg):
            gsm8k_rope.debug(provided_kwargs.get(keyword, NotProvidedKwarg()), )
            kwargs[keyword] = provided_kwargs[keyword]
    return kwargs


def auto_apply_dynamic_kwargs(func, **provided_kwargs):
    gsm8k_rope.debug("In Auto Apply Dynamic Kwargs")
    sig = inspect.signature(func)
    names = list(name for name, param in sig.parameters.items())

    kwargs = dynamic_kwargs(names, **provided_kwargs)
    signature_str = '\n'.join( f"Name: {name}, Kind: {param.kind}, Default: {param.default}" for name, param in sig.parameters.items() )

    required_kwargs = []
    has_var_keyword = False
    for name in names:
        if sig.parameters[name].default == inspect._empty and (sig.parameters[name].kind & 1): # required positional_or_keyword (1) or keyword_only (3)
            required_kwargs.append(name)
        if sig.parameters[name].kind == 4:  # var_keyword
            has_var_keyword = True
    # we do not accept kind==2 var_positional


    # inspect the kwargs
    for name in names:
        if name not in kwargs and name in required_kwargs:
            raise ValueError(f"KwargError: kwarg not provided: {name}\nkwargs provided: {provided_kwargs}\nrequired: {required_kwargs}\nSignature:\n{signature_str}")

    for name in provided_kwargs:
        if name not in names and not has_var_keyword:
            raise ValueError(f"KwargError: Unknown kwarg: {name}\nkwargs provided: {provided_kwargs}\nrequired: {required_kwargs}\nSignature:\n{signature_str}")
    gsm8k_rope.debug(f"[Dynamic] Provided kwargs: {provided_kwargs} \n          Applied kwargs: {kwargs}")
    return func(**kwargs)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set_running_specs", methods=["POST"])
def set_running_specs():
    dynamic_running_specs = request.get_json(force=True)
    try:
        manager.set_kwargs_for_running_experiment(**dynamic_running_specs)
        return jsonify({"message": "success"})
    except Exception as e:
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500


@app.route("/run_all")
def run_all():
    def sse_wrapping():
        for result in manager.run_all_experiments(True):
            yield f"data: {json.dumps(result)}\n\n"  # SSE follows the format `data: ...\n\n`

    try:
        return Response(sse_wrapping(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500

@app.route("/run")
def run():
    kwargs = {key: value for key, value in request.args.items()}
    for key in kwargs:
        print(key, kwargs[key])
        if key.startswith("dynamic_model_info_list"):
            print("!")
            kwargs[key] = json.loads(kwargs[key])

    exp_dir = kwargs.get("exp_dir", None)
    if exp_dir is None:
        return jsonify({"error": f"exp dir not provided"})
    kwargs.pop("exp_dir")
    experiment = manager.open_experiment(exp_dir=exp_dir)
    print(f"RUN the experiment on {exp_dir}")
    print(kwargs)
    print()

    @stream_with_context
    def sse_wrapping():

        stop_event = threading.Event()
        try:
            for result in manager.run_one_experiment(experiment, True, stop_event=stop_event, **kwargs):

                if request.environ.get('wsgi.input').closed:
                    print("Client disconnected.")

                    break
                yield f"data: {json.dumps(result)}\n\n"  # SSE follows the format `data: ...\n\n`
        except GeneratorExit:
            print("Generator Exit!!!!!")


    try:
        response = Response(sse_wrapping(), mimetype="text/event-stream")
        print("End of response")
        return response
    except Exception as e:
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500

@app.route("/init")
def init():
    try:

        return jsonify({})
    except Exception as e:
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500

@app.route("/nodelist")
def nodelist():
    try:
        # print("nodelist", manager)
        nodelist = manager.get_nodelist_with_status()

        return jsonify(nodelist)
    except Exception as e:
        raise e
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500


@app.route("/kill_all_jobs")
def kill_all_jobs():
    # try:
        print("Kill")
        manager.kill_backend_jobs()
        return jsonify({})
    # except Exception as e:
    #     return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500


@app.route("/submit_jobs", methods=["POST"])
def submit_jobs():
    kwargs = request.get_json(force=True)
    try:
        for x in manager.submit_backend_jobs(**kwargs):
            continue
        return jsonify({})
    except Exception as e:
        print("????", e)
        raise e
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500


@app.route("/get_history")
def get_history():
    try:
        return jsonify(manager.get_history())
    except Exception as e:
        print("????", e)
        raise e


# @app.route("/set_the_results_library", methods=["POST"])
# def set_the_results_library():
#     kwargs = request.get_json(force=True)
#     try:
#         auto_apply_dynamic_kwargs(manager.set_the_results_library, **kwargs)
#         # manager.set_the_results_library(**dynamic_kwargs(["directory", "save_to_config"], **kwargs))
#     except Exception as e:
#         return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 500
@app.route("/submit_command", methods=["POST"])
def submit_command():
    kwargs = request.get_json(force=True)
    try:
        completed_process = subprocess.run(kwargs["command"].split(), capture_output=True)
        return jsonify({"stdout": completed_process.stdout.decode("utf-8"), "stderr": completed_process.stderr.decode("utf-8")})
    except FileExistsError as e:
        return jsonify({"message": "failed", "error": f"Exception of type {type(e)}: {e}"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400


@app.route("/define_experiment", methods=["POST"])
def define_experiment():
    kwargs = request.get_json(force=True)
    try:
        gsm8k_rope.debug("In define_experiment", kwargs)
        auto_apply_dynamic_kwargs(manager.define_experiment, **kwargs)
        return jsonify({"message": "success"})
    except FileExistsError as e:
        return jsonify({"message": "failed", "error": f"Exception of type {type(e)}: {e}"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/update_api_key", methods=["POST"])
def update_api_key():
    kwargs = request.get_json(force=True)
    try:
        manager.update_api_key(kwargs["api_name"], kwargs["api_key"])
        return jsonify({"message": "success"})
    except FileExistsError as e:
        return jsonify({"message": "failed", "error": f"Exception of type {type(e)}: {e}"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/get_api_keys", )
def get_api_keys():
    try:
        return jsonify({"api_keys": manager.get_api_keys()})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400


@app.route("/find_and_open_all_experiments_in_dir")
def find_and_open_all_experiments_in_dir():
    # try:
        experiment_list = manager.find_and_open_all_experiments_in_dir()
        for experiment in experiment_list:
            gsm8k_rope.debug(experiment.find_existing_tests(False))
        experiment_dir_list = [{"path": experiment.directory_full_path(),
                                "name": experiment.name,
                                "type": experiment.experiment_type,
                                "version": experiment.version,
                                "tester": experiment.linking_tester_class.__name__,
                                "specs": experiment.get_experiment_specs().to_dict(),
                                "specs_expansion": [{"specs": specs.to_dict(),
                                                     "existing": experiment.existing_directory_for_specs(specs,
                                                                                                         ["status"])
                                                     } for specs in experiment.get_experiment_specs().expand()],
                                "default_test_specs": experiment.get_running_specs(),
                                "existing_tests": [{
                                        "path": things["path"],
                                        "mtime": datetime.fromtimestamp(os.path.getmtime(things["path"])).isoformat(),
                                        "ctime": datetime.fromtimestamp(os.path.getctime(things["path"])).isoformat(),
                                        "specs": {item.split("=")[0]: item.split("=")[1] for item in specs.split(",")},
                                        "status": things["status"],
                                    } for specs, things in experiment.find_existing_tests(False).items()
                                ]} for experiment in experiment_list]
        gsm8k_rope.debug(experiment_dir_list)
        return jsonify(experiment_dir_list)
    # except Exception as e:
    #     print(e)
    #     return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/import_slurm_settings", methods=["POST"])
def import_slurm_settings():
    kwargs = request.get_json(force=True)
    try:
        path = kwargs["filename"]
        path = os.path.expanduser(path)
        settings = manager.import_slurm_settings_from_file(path)
        return jsonify({"message": "success", "settings": settings})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/manage_auto_submit_job", methods=["POST"])
def manage_auto_submit():
    kwargs = request.get_json(force=True)
    try:
        manager.manage_auto_submit_job(kwargs["mode"], kwargs["nodes"])
        return jsonify({"message": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/submit_slurm_job", methods=["POST"])
def submit_slurm_job():
    kwargs = request.get_json(force=True)
    try:
        manager.submit_slurm_job(**kwargs)
        return jsonify({"message": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/cancel_slurm_job", methods=["POST"])
def cancel_slurm_job():
    kwargs = request.get_json(force=True)
    try:
        manager.cancel_slurm_job(kwargs["jobid"])
        return jsonify({"message": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/get_slurm_settings")
def get_slurm_settings():
    try:
        return jsonify({"settings": manager.get_slurm_settings()})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400


@app.route("/get_user_info")
def get_user_info():
    try:

        return jsonify(manager.get_user_info())
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400

@app.route("/slurm_jobs")
def get_slurm_jobs():
    try:
        jobs = manager.get_slurm_jobs()
        return jsonify(jobs)
    except Exception as e:
        print(e)
        return jsonify({"error": f"Exception of type {type(e)}: {e}"}), 400


"""
    def start_server(self, port="1942"):
        completed_process = subprocess.run(["lsof", "-i", f"tcp:{port}"] , capture_output=True)
        output = completed_process.stdout.decode("utf-8").strip()
        if len(output):
            raise ValueError(f"Port {port} is used by the following processes:\n{output}")
        self.web_server ="""
if __name__ == '__main__':
    import logging
    # logging.getLogger('werkzeug').disabled = True
    # app.logger.disabled = True
    gsm8k_rope.output_level = "debug"
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=True, threaded=True)
