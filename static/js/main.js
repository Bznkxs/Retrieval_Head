function parseColor(text_str) {
    let text_str_pure = text_str.replace(/\x1B.*?m/g, "")
    let first_match = text_str.match(/\x1B.*?m/g)
    if (first_match !== null) {
        first_match = first_match[0];
        console.log(first_match);
        if (first_match.match(/\x1B\[38;2;/g)) {
            // rgb
            first_match = first_match.replace(/\x1B\[38;2;/g, "")
            first_match = first_match.replace(/m/, "")
            let rgb = first_match.split(";");
            let r = parseInt(rgb[0]), g = parseInt(rgb[1]), b = parseInt(rgb[2]);
            return {text: text_str_pure, color: `rgb(${r},${g},${b})`};
        } else if (first_match.match(/93m/)) {
            return {text: text_str_pure, color: "rgb(255,165,0)"}
        } else if (first_match.match(/32m/)) {
            return {text: text_str_pure, color: "green"};
        } else {
            return {text: text_str_pure, color: "black"}
        }
    } else {
        return {text: text_str_pure, color: "black"};
    }
}



document.addEventListener('DOMContentLoaded', () => {
    const experiment_list_list = document.getElementById("experiment_list_list");
    const node_list_list = document.getElementById("node_list_list");
    const new_experiment_name_input = document.getElementById("new-experiment-name-input");
    const new_experiment_type_input = document.getElementById("new-experiment-type-input");
    const new_experiment_version_input = document.getElementById("new-experiment-version-input");
    const new_experiment_tester_input = document.getElementById("new-experiment-tester-input");
    const new_experiment_specs_input = document.getElementById("new-experiment-specs-input");
    const new_experiment_running_specs_input = document.getElementById("new-experiment-running-specs-input");
    const auto_management_button = document.getElementById("button-auto-management");
    const megatron_list_of_argument_input = document.getElementById("megatron_list_of_argument_input");
    megatron_list_of_argument_input.innerText = '{                           "model_path":"/work/nvme/bdtq/yufengd4/llama_3.1_mg",\n' +
        '                           "tokenizer_path":"/work/nvme/bdtq/mtian8/models/HF_model/Llama-3.1-8B-Instruct",\n' +
        '                           "rope_base":"500000",\n' +
        '                           "model_type":"llama3",\n' +
        '                           "attn_impl":"te"}'
    let auto_job_mode = "unknown";
    let auto_job_nodes = "0"
    let target_job_mode = "uninitiated"
    fetch("/init").then(async () => {
        async function refresh_experiment_list() {
            const response = await fetch('/find_and_open_all_experiments_in_dir');
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
                console.log(data)
                return;
            }
            experiment_list_list.replaceChildren();  // remove all children
            for (let i = 0; i < data.length; ++i) {
                const newDiv = document.createElement("div");

                experiment_list_list.appendChild(newDiv);
                const pathDiv = document.createElement("div");
                pathDiv.innerText = data[i].path;
                newDiv.appendChild(pathDiv);
                const showContentDiv = document.createElement("div");
                newDiv.appendChild(showContentDiv);
                showContentDiv.style.display = "none";
                pathDiv.addEventListener("click", () => {
                    if (showContentDiv.style.display === "none") {
                        showContentDiv.style.display = "inherit";
                    } else {
                        showContentDiv.style.display = "none";
                    }
                })
                pathDiv.style.cursor = "pointer";
                const runExperimentButton = document.createElement("div");
                showContentDiv.appendChild(runExperimentButton);
                runExperimentButton.innerText = ">Run Experiment<"
                runExperimentButton.classList.add("btn");

                const modifySettingsButton = document.createElement("div")
                showContentDiv.appendChild(modifySettingsButton);
                modifySettingsButton.innerText = ">Edit Experiment Settings (Specs Will be Imported to Inputs Below)<"
                modifySettingsButton.classList.add("btn");
                modifySettingsButton.addEventListener("click", () => {
                    new_experiment_name_input.value = data[i].name;
                    new_experiment_type_input.value = data[i].type;
                    new_experiment_version_input.value = data[i].version;
                    new_experiment_tester_input.value = data[i].tester;
                    new_experiment_specs_input.value = JSON.stringify(data[i].specs);
                    new_experiment_running_specs_input.value = JSON.stringify(data[i].default_test_specs)

                })

                const specs = data[i].specs;
                const constantSpecs = {};


                const specsOuterDiv = document.createElement("div");
                showContentDiv.appendChild(specsOuterDiv);
                const constantSpecsDiv = document.createElement("div");
                specsOuterDiv.appendChild(constantSpecsDiv);
                const specsDiv =  document.createElement("div");
                specsOuterDiv.appendChild(specsDiv);
                specsDiv.style.display = "grid";


                for (let spec in specs) {
                    if (specs[spec].choices && specs[spec].choices.length === 1) {
                        constantSpecs[spec] = specs[spec].choices[0]
                    }
                }

                constantSpecsDiv.innerHTML = "<b>Constant Specs:</b>" + JSON.stringify(constantSpecs);

                const specs_col = Object.keys(specs);
                specs_col.sort();

                const augmented_specs_col = [];
                for (const spec in specs_col) {
                    if (specs_col[spec] in constantSpecs) {
                        continue;
                    }
                    augmented_specs_col.push(specs_col[spec]);
                }
                augmented_specs_col.push("status");
                const specs_status_divs = [];
                for (const spec_idx in augmented_specs_col) {
                    const spec = augmented_specs_col[spec_idx];
                    const gridDiv = document.createElement("div");
                    specsDiv.appendChild(gridDiv);
                    gridDiv.innerText = spec;
                    gridDiv.classList.add("spec_head");
                    console.log(spec);
                }
                specsDiv.style.gridTemplateColumns = "repeat(" + (augmented_specs_col.length).toString() + ", 1fr)";
                for (let specs_idx in data[i].specs_expansion) {
                    let specs = data[i].specs_expansion[specs_idx].specs;
                    for (let spec in specs) {
                        if (spec in constantSpecs) {
                            continue;
                        }
                        const gridDiv = document.createElement("div");
                        specsDiv.appendChild(gridDiv);
                        gridDiv.innerText = specs[spec];
                        console.log(specs[spec]);
                    }
                    const gridDiv = document.createElement("div");
                    specsDiv.appendChild(gridDiv);
                    if (data[i].specs_expansion[specs_idx].existing) {
                        let {text, color} = parseColor(data[i].specs_expansion[specs_idx].existing.status);
                        gridDiv.style.color = color;
                        gridDiv.innerText = text;
                        if (text.match("Running")) {
                            runExperiment().then();
                        }
                    }
                    else {
                        gridDiv.innerText = "nonexistent";
                    }


                    specs_status_divs.push(gridDiv);
                }
                async function runExperiment () {
                    let args = `exp_dir=${data[i].path}`;
                    console.log("APIKEYS")
                    console.log(apiKeys)
                    console.log(Number(document.getElementById("num_threads-input")))
                    for (let key in apiKeys) {
                        args += `&dynamic_model_info_list_for_${key}=[[`
                        for (let i = 0; i < Number(document.getElementById("num_threads-input").value); ++i) {
                            if (i > 0) {
                                args += ",";
                            }
                            args += '"' + apiKeys[key] + '"';

                        }
                        args += "]]";
                    }
                    console.log(args)

                    const source = new EventSource(`/run?${args}`);
                    source.onmessage = function(event) {
                        const eventData = JSON.parse(event.data);
                        if (eventData.status === "running") {
                            console.log("Running")
                            console.log(eventData.augmented_all_specs)
                            const augmented_all_specs = eventData.augmented_all_specs;
                            for (let specs_idx in data[i].specs_expansion) {
                                let specs = data[i].specs_expansion[specs_idx].specs;
                                console.log("> specs")
                                console.log(specs)
                                for (let data_specs_idx in augmented_all_specs) {
                                    let same_flag = true;
                                    let data_specs = augmented_all_specs[data_specs_idx];
                                    for (let spec in specs) {
                                        if (spec !== "status" && data_specs[spec] !== specs[spec]) {
                                            same_flag = false;
                                            break;
                                        }
                                    }

                                    if (same_flag) {
                                        let {text, color} = parseColor(data_specs["status"]);
                                        specs_status_divs[specs_idx].style.color = color;
                                        specs_status_divs[specs_idx].innerText = text;
                                    }
                                }
                            }
                        }
                        if (eventData.status === "experiment finished") {
                            source.close();
                        }
                    }
                }
                runExperimentButton.addEventListener("click", runExperiment)

                const existingTestsDiv = document.createElement("div");
                showContentDiv.appendChild(existingTestsDiv);
                for (let j = 0; j < data[i].existing_tests.length; ++j) {
                    const testDiv = document.createElement("div");
                    testDiv.style.whiteSpace = "pre";
                    testDiv.innerText = "    " + data[i].existing_tests[j].path + " | Last Modified: " + data[i].existing_tests[j].mtime;
                    existingTestsDiv.appendChild(testDiv);
                }
            }

        }

        let showInfo = {};
        async function refresh_node_list() {
            const response = await fetch("/nodelist");
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
                console.log(data)
                return;
            }
            // console.log(data)
            node_list_list.replaceChildren();
            let node_set = new Set();
            let port_set = new Set();


            for (let node_port in data) {
                const parts = node_port.split(":");
                const node = parts[0];
                const port = parts[1];
                node_set.add(node);
                port_set.add(port);
            }

            let node_list = Array.from(node_set);
            node_list.sort();
            let port_list = Array.from(port_set);
            port_list.sort();
            node_list_list.style.display = "grid";
            node_list_list.style.gridTemplateColumns = `repeat(${port_list.length + 1}, 1fr)`;
            const gridDiv = document.createElement("div")
            gridDiv.innerText = "Nodes\\Ports"
            gridDiv.classList.add("spec_head");
            node_list_list.appendChild(gridDiv);
            for (let j in port_list) {
                    const gridDiv = document.createElement("div")
                    gridDiv.innerText = port_list[j]
                    gridDiv.classList.add("spec_head");
                    node_list_list.appendChild(gridDiv);
                }
            for (let i in node_list) {
                const gridDiv = document.createElement("div")
                gridDiv.innerText = node_list[i]
                gridDiv.classList.add("spec_head");
                node_list_list.appendChild(gridDiv)
                for (let j in port_list) {
                    const gridDiv = document.createElement("div")
                    const readable = data[node_list[i] + ":" + port_list[j]].readable;

                    gridDiv.style.color = data[node_list[i] + ":" + port_list[j]].display_color
                    const info = data[node_list[i] + ":" + port_list[j]].info
                    if (showInfo[node_list[i] + ":" + port_list[j]] === undefined) {
                        showInfo[node_list[i] + ":" + port_list[j]] = false;
                    }

                    function updateDisplay() {
                        if (showInfo[node_list[i] + ":" + port_list[j]] && info) {
                            gridDiv.innerText = JSON.stringify(info);
                        } else {
                            gridDiv.innerText = readable
                        }
                    }
                    updateDisplay();
                    gridDiv.addEventListener("click", ()=>{
                        showInfo[node_list[i] + ":" + port_list[j]] = !showInfo[node_list[i] + ":" + port_list[j]];
                        updateDisplay();
                    })

                    if (gridDiv.style.color === "yellow") {
                        gridDiv.style.color = "brown";
                    }
                    node_list_list.appendChild(gridDiv);



                }
            }

        }

        const slurm_job_operation_button_texts = {}

        async function get_slurm_settings() {
            const response = await fetch("/get_slurm_settings");
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
                console.log(data)
            }
            document.getElementById("new-job-settings-input").value = JSON.stringify(data.settings);
        }

        get_slurm_settings().then(()=>{});

        async function refresh_slurm_job_list() {
            const response = await fetch("/slurm_jobs");
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
                console.log(data)
            }
            console.log(data)
            const job_list_list = document.getElementById("job_list_list");
            job_list_list.replaceChildren();

            auto_job_mode = data.auto_submit.mode
            auto_job_nodes = data.auto_submit.nodes
            if (!auto_job_nodes) {
                auto_job_nodes = "0";
            }

            const current_nodes = data.nodes_cnt
            document.getElementById("nodes_cnt_div").innerText = `Submitted jobs contain ${current_nodes} nodes in total.`
            set_auto_management_button_inner_text()

            if (data.data.length === 0) {
                job_list_list.innerText = "You haven't submitted any job right now."
                return;
            }

            job_list_list.style.display = "grid";
            job_list_list.style.gridTemplateColumns = `repeat(${data.headers.length + 1}, 1fr)`;
            for (let h in data.headers) {
                const gridDiv = document.createElement("div")
                gridDiv.innerText = data.headers[h];
                gridDiv.classList.add("spec_head");
                job_list_list.appendChild(gridDiv);
            }
            const gridDiv = document.createElement("div")
            gridDiv.innerText = "Operation";
            gridDiv.classList.add("spec_head");
            job_list_list.appendChild(gridDiv);
            const job_id_idx = data.headers.indexOf("JOBID");

            let current_jobs = [];
            for (let i in data.data) {
                const jobid = data.data[i][job_id_idx];
                current_jobs.push(jobid);
            }
            let oldjobs = Object.keys(slurm_job_operation_button_texts)
            for (let job in oldjobs) {
                if (!current_jobs.includes(job)) {
                    delete slurm_job_operation_button_texts[job]
                }
            }
            for (let i in data.data) {
                for (let j in data.data[i]) {
                    const gridDiv = document.createElement("div")
                    gridDiv.innerText = data.data[i][j];
                    job_list_list.appendChild(gridDiv);
                }
                const jobid = data.data[i][job_id_idx];
                const gridDiv = document.createElement("div")
                if (slurm_job_operation_button_texts[jobid]) {
                    gridDiv.innerText = slurm_job_operation_button_texts[jobid];
                } else {
                    gridDiv.innerText = `Cancel JOBID=${jobid}`;
                    slurm_job_operation_button_texts[jobid] = gridDiv.innerText;
                }

                job_list_list.appendChild(gridDiv);
                gridDiv.classList.add("btn");
                gridDiv.addEventListener("mouseup", (event) => {
                    if (event.button === 0) {
                        if (gridDiv.innerText === `Cancel JOBID=${jobid}`) {
                            gridDiv.innerText = "Cancel (Left Click Once More to Confirm)"

                        } else if (gridDiv.innerText === "Cancel (Left Click Once More to Confirm)") {
                            gridDiv.innerText = "Canceling"
                            gridDiv.classList.remove("btn")
                            fetch("/cancel_slurm_job",  {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({ jobid: jobid })
                            });
                        }
                    } else {
                        gridDiv.innerText = `Cancel JOBID=${jobid}`;
                    }
                    slurm_job_operation_button_texts[jobid] = gridDiv.innerText
                })
                gridDiv.addEventListener("mouseleave", () => {
                    gridDiv.innerText = `Cancel JOBID=${jobid}`;
                    slurm_job_operation_button_texts[jobid] = gridDiv.innerText;
                })
            }
            console.log(data)


        }

        async function get_user_info() {
            const response = await fetch("/get_user_info");
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
                console.log(data)
            }
            document.getElementById("user-info-h2").innerText = `Hello Manager ${data.username}!`
        }

        get_user_info()


        function set_auto_management_button_inner_text() {
            if (auto_job_mode === "on" && (target_job_mode === "on" || target_job_mode === "uninitiated")) {
                auto_management_button.innerText = `Current Status: On. Keeping at least ${auto_job_nodes} nodes. Click to Turn Off Auto Management`
            } else if (auto_job_mode === "off" && (target_job_mode === "off" || target_job_mode === "uninitiated")) {
                auto_management_button.innerText = "Current Status: Off. Click to Turn On Auto Management"
            } else {
                auto_management_button.innerText = `Current Status: ${auto_job_mode}. Target Status: ${target_job_mode}. No operations permitted`
            }
        }

        auto_management_button.addEventListener("click", async () => {
            target_job_mode = ""
            if (auto_job_mode === "on") {
                target_job_mode = "off";
            }
            if (auto_job_mode === "off") {
                target_job_mode = "on";
            }
            if (target_job_mode === "") {
                return;
            }
            auto_job_mode = "unknown";
            set_auto_management_button_inner_text();
            const response = await fetch("/manage_auto_submit_job", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    mode: target_job_mode,
                    nodes: document.getElementById("auto-management-nodes-input").value
                })
            })
            const data = await response.json();
            console.log(response)
            document.getElementById("operation-status-div").innerText = `Operation Manage Auto Submit returned status ${response.status}. Response: ${JSON.stringify(data)}`

        })

        refresh_experiment_list().then(() => {});


        async function keep_refreshing_node_list() {
            while (true) {
                refresh_node_list().then(() => {});
                refresh_slurm_job_list().then(() => {})
                await new Promise(resolve => setTimeout(resolve, 4000));
            }
        }

        keep_refreshing_node_list().then(() => {});


        document.getElementById('button-refresh').addEventListener('click', async () => {
            refresh_experiment_list().then(() => {});
            refresh_node_list().then(() => {});
        });
        async function new_experiment() {
            console.log(new_experiment_specs_input.value)
            console.log(new_experiment_running_specs_input.value)
            const response = await fetch("/define_experiment", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    experiment_name: new_experiment_name_input.value,
                          experiment_version: new_experiment_version_input.value,
                          experiment_type: new_experiment_type_input.value,
                          linking_tester_class: new_experiment_tester_input.value,
                          experiment_specs_dict: JSON.parse(new_experiment_specs_input.value),
                          default_test_specs_dict: JSON.parse(new_experiment_running_specs_input.value),
                })
            })
            const data = await response.json();
            if (data.message === "success") {
                await refresh_experiment_list();
            } else {
                console.log(data)
            }
        }

        document.getElementById("button-new-experiment").addEventListener("click", async() => {
            new_experiment().then(() => {});
        })

        document.getElementById("button-kill-all-megatron").addEventListener("click", async() => {
            const response = await fetch("/kill_all_jobs")
        })

        document.getElementById("button-submit-megatron").addEventListener("click", async() => {
            const response = await fetch("/submit_jobs", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: megatron_list_of_argument_input.value
            })
            getHistory();
        })
        getHistory();
        async function getHistory() {
            const response = await fetch("/get_history");
            if (response.status !== 200) {
                return undefined;
            }
            const data = await response.json();
            console.log(data);
            console.log("____")
            // data.history.submit_backend_jobs_history
            const megatron_history_list_div = document.getElementById("megatron_history_list_div");
            megatron_history_list_div.replaceChildren()
            for (let idx in data.submit_backend_jobs_history) {
                const submit_backend_history = data.submit_backend_jobs_history[idx];
                const div = document.createElement("div");
                megatron_history_list_div.appendChild(div);
                div.innerText = JSON.stringify(submit_backend_history);
                div.addEventListener("click", () => {
                    megatron_list_of_argument_input.value = JSON.stringify(submit_backend_history);
                })
                div.classList.add("btn")
                megatron_history_list_div.appendChild(div)
            }

            return data;
        }



        document.getElementById("button-submit-job").addEventListener("click", async() => {
            const response = await fetch("/submit_slurm_job", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: document.getElementById("new-job-settings-input").value
            })
            const data = await response.json();
            console.log(response)
            document.getElementById("operation-status-div").innerText = `Operation Submit Slurm Job returned status ${response.status}. Response: ${JSON.stringify(data)}`
        })

        document.getElementById("button-import-setting").addEventListener("click", async() => {
            const response = await fetch("/import_slurm_settings", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    filename: document.getElementById("sbatch-script-file-input").value
                })
            })
            if (response.status !== 200) {
                console.log(response)
            }
            const data = await response.json();
            console.log(response)
            if (data.settings)
                document.getElementById("new-job-settings-input").value = JSON.stringify(data.settings);
            document.getElementById("operation-status-div").innerText = `Operation Import returned status ${response.status}. Response: ${JSON.stringify(data)}`
        })

        let apiKeys = null
        async function getAPIKeys() {
            const response = await fetch("/get_api_keys")
            const data = await response.json();
            console.log(response)
            if (response.status === 200) {
                apiKeys = data.api_keys;
                document.getElementById("api-keys-list").replaceChildren();
                for (let key in data.api_keys) {
                    const div = document.createElement("div");
                    div.innerText = `${key}: ${data.api_keys[key] ? 'Provided': 'None'}`
                    document.getElementById("api-keys-list").appendChild(div)
                }
            } else {
                document.getElementById("operation-status-div").innerText = `Operation Get API Keys returned status ${response.status}. Response: ${JSON.stringify(data)}`
            }
        }

        getAPIKeys()

        async function uploadAPIKey() {
            const response = await fetch("/update_api_key", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    api_name: document.getElementById("keyname-input").value,
                    api_key: document.getElementById("keyvalue-input").value
                })

            })
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
            }
            document.getElementById("operation-status-div").innerText = `Operation uploadAPIKey returned status ${response.status}. Response: ${JSON.stringify(data)}`
            return response;
        }

        document.getElementById("refresh-keys-button").addEventListener("click", async() => {
            getAPIKeys();
        })

        document.getElementById("submit-key-button").addEventListener("click", async() => {
            const response = await uploadAPIKey();
            if (response.status === 200) {
                getAPIKeys();
            }
            document.getElementById("operation-status-div").innerText = `Operation uploadAPIKey returned status ${response.status}. `

        })


        document.getElementById("button-submit-command").addEventListener("click", async() => {
            const response = await fetch("/submit_command", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    command: document.getElementById("command-input").value
                })
            })
            const data = await response.json();
            if (response.status !== 200) {
                console.log(response)
            } else {
                document.getElementById("stdout-output-div").innerText = data.stdout;
                document.getElementById("stderr-output-div").innerText = data.stderr;
                document.getElementById("stdout-output-div").style.whiteSpace = "pre";
                document.getElementById("stderr-output-div").style.whiteSpace = "pre";

            }


            console.log(response)
            document.getElementById("operation-status-div").innerText = `Operation Submit Command returned status ${response.status}. Response: ${JSON.stringify(data)}`
        })
    });

});

