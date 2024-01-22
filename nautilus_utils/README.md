# Nautilus Tutorial

**Kubernetes Pod:** Group of containers that are deployed together on the same host. If you frequently deploy single-container pods, you can generally replace the word "pod" with "container".

**Kubernetes Job:** A Job is a daemon which watches your pod and makes sure it exited with exit status 0. If it did not for any reason, it will be restarted up to `backoffLimit` number of times.

## [POLICIES](https://docs.nationalresearchplatform.org/userdocs/start/policies/)

- Use **job** to run batch jobs and set RIGHT resources request. 
    - DO NOT run jobs with "sleep" command or equivalent (script ending with "sleep"), otherwise you will be banned from nautilus permanently. 

    - Running jobs in manual mode (sleep infinity command and manual start of computation) is prohibited, and user can be banned for wasting GPU resources.

- Use **deployment** for long-running pods and SET MINIMAL resources request

- Use **pods** BUT it will be destroyed in 6 hrs

- Avoid wasting resources: if you've requested some CPU/GPU resources, use it, and free up once computation is done.

- When you request GPUs for your pod, **nobody else can use those until you stop your pod**. Always delete your pod when your computation is done to let other users use the GPUs. You should only schedule GPUs that you can actually use. 

- The only reason to **request more than a single GPU** is when your GPU utilization is close to 100% and you can leverage more GPUs.


## Memory allocation

- A **request** is what will be reserved for your pod on a node for scheduling purposes. 

- A **limit** is the maximum which your pod should never exceed. 

- If pod goes over its memory limit, it will be KILLED. 

- While it's important to set the limit properly, it's also important to not set the request too high. Your request should be as close as possible to the average resources you're going to consume, and limit should be a little higher than the highest peak you're expecting to have


## Interactive Use vs Batch

- There are so called operators to control the behaviour of pods. Since pods don't stop themselves in normal conditions, and don't recover in case of node failure, we assume every pod running in the system without any controller to be interactive -- started for a short period of time for active development / debugging. 

- We limit pods to request a maximum of 2 GPUs, 32 GB RAM and 16 CPU cores which will be destroyed in 6 hours, unless you request an exception for your namespace (in case you run jupyterhub or some other application controlling the pods for you).

- If you need to run a larger and longer computation, you need to use one of available Workload Controllers. We recommend running those as Jobs - this will closely watch your workload and make sure it ran to completion (exited with 0 status), shut it down to free up the resources, and restart if node was rebooted or something else happened.

- In case you need some pod to run idle for a long time, you can use the Deployment controller. Make sure you set minimal request and proper limits for those to get the Burstable QoS


## Running batch jobs

- Highly recommended to use Jobs for any kind of development and computations. 

- This ensures you never lose your work, get the results in the most convenient way, and don't waste resources, since this method does not require any babysitting of processes from you. 

- Once your development is done, you are immediately ready to run a large-scale stuff with no changes to the code and minimal changes in the definition, plus your changes are saved in Git.

- **Since jobs in Nautilus can run forever, you can only run jobs with meaningful `command` field.**

- When job is finished, your pod will stay in Completed state, and Job will have `COMPLETIONS` field 1/1. For long jobs, the pods can have Error, Evicted, and other states until they finish properly or `backoffLimit` is exhausted.

- Retries: `backoffLimit` specifies how many times pod will run in case the exit status of your script is not 0 or if pod was terminated for a different reason (for example a node was rebooted). Good idea to keep it more than 0.

- You can group several commands and use pipes:
```
command:
  - sh
  - -c
  - "cd /home/user/my_folder && apt-get install -y wget && wget pull some_file && do something else"
```


## Tensorboard

- If you are training ML models using Python, Tensorflow, PyTorch, etc. it is common to plot real time statistics to Tensorboard. 

- You should first activate the tensorboard in the pods
```
tensorboard --logdir=${LOG-FILE}
```

- kubectl can link your local port to the specified port of the pods
```
kubectl port-forward ${POD_NAME} ${REMOTE-PORTNUM}:${LOCAL-PORTNUM}
```

- Tensorboard website can be seen at [http://localhost:LOCAL-PORTNUM]()


## Available GPUs and choosing GPU type

More detailed info can be found [here](https://docs.nationalresearchplatform.org/userdocs/running/gpu-pods/#:~:text=If%20you%20need%20more%20graphical%20memory%2C%20use%20this%20table%20or%20official%20specs%20to%20choose%20the%20type%3A).

- Quadro-M4000 (8G)
- NVIDIA-GeForce-GTX-1070 (8G)
- NVIDIA-GeForce-GTX-1080 (8G)
- NVIDIA-A100-PCIE-40GB-MIG-2g.10gb	(10G)
- NVIDIA-GeForce-GTX-1080-Ti (12G)
- NVIDIA-GeForce-RTX-2080-Ti (12G)
- NVIDIA-A10 (24G)
- NVIDIA-GeForce-RTX-3090 (24G)
- NVIDIA A100 80GB PCIe MIG 1g.10gb

Below are higher specs GPUs (your pods will only be scheduled on these if you request the type explicitly):

- NVIDIA-TITAN-RTX (24G)
- NVIDIA-RTX-A5000 (24G)
- Quadro-RTX-6000 (24G)
- Tesla-V100-SXM2-32GB (32G)
- NVIDIA-A40 (48G)
- NVIDIA-RTX-A6000 (48G)
- Quadro-RTX-8000 (48G)
- NVIDIA A100-SXM4-80GB (consumes much more memory space compared to other GPUs)


## Kubernetes commands
Below are some commonly used Kubernetes commands along with brief descriptions:

Get Information about Pods or Jobs
```
kubectl get pods
kubectl get jobs
```

Fetch information about the cluster nodes, specifically looking for nodes with NVIDIA GPUs and their corresponding GPU models.
```
kubectl get nodes -L nvidia.com/gpu.product
```

Create a new Kubernetes job based on the configuration provided. Jobs are used for running batch processes or scheduled tasks.
```
kubectl create -f /path/to/yaml/config/file
```

Execute a command interactively within a specific pod
```
kubectl exec -it <pod_name> -- /bin/bash
```

Delete a specific pod
```
kubectl delete pod <pod_name>
```

Deletes a specific job which terminates its associated pods and frees up any allocated resources
```
kubectl delete job <job_name>
```

Retrieve information about Persistent Volume Claims (PVCs) in the current namespace. Displays details such as PVC names, statuses, and storage capacity. PVCs are used to request persistent storage in a Kubernetes cluster.
```
kubectl get pvc
```

Show stdout and stderr output
```
kubectl logs <pod_name>
```

To make sure you did everything correctly after submitting the job, look at the corresponding pod yaml and check the resulting nodeAffinity is as expected
```
kubectl get pod ... -o yaml
```
