# Nautilus Tutorial

## POLICIES

- A Job is a daemon which watches your pod and makes sure it exited with exit status 0. If it did not for any reason, it will be restarted up to `backoffLimit` number of times.

- Use jobs with actual script instead of sleep whenever possible to ensure your pod is not wasting GPU time

<!-- - To make sure you did everything correctly after you've submited the job, look at the corresponding pod yaml (`kubectl get pod ... -o yaml`) and check that resulting nodeAffinity is as expected -->

- Use **job** to run batch jobs and set RIGHT resources request. 
    - DO NOT run jobs with "sleep" command or equivalent (script ending with "sleep"), otherwise you will be banned from nautilus permanently (trust me, they do impose ban). 
    - Running in manual mode (sleep infinity command and manual start of computation) is prohibited, and user can be banned.

- Use **deployment** for long-running pods and SET MINIMAL resources request

- Use **pods** BUT it will be destroyed in 6 hrs

- Avoid wasting resources: if you've requested something, use it, and free up once computation is done.

- When you request GPUs for your pod, **nobody else can use those until you stop your pod**. Always delete your pod when your computation is done to let other users use the GPUs

- You should only schedule GPUs that you can actually use. 

- The only reason to **request more than a single GPU** is when your GPU utilization is close to 100% and you can leverage more.

### Memory allocation

- A **request** is what will be reserved for your pod on a node for scheduling purposes. A **limit** is the maximum which your pod should never exceed. 

- If pod goes over its memory limit, it WILL BE KILLED. 

- While it's important to set the Limit properly, it's also important to not set the Request too high. Your request should be as close as possible to the average resources you're going to consume, and limit should be a little higher than the highest peak you're expecting to have


### Interactive Use vs Batch

- There are so called operators to control the behaviour of pods. Since pods don't stop themselves in normal conditions, and don't recover in case of node failure, we assume every pod running in the system without any controller to be interactive -- started for a short period of time for active development / debugging. 

- We limit those to request a maximum of 2 GPUs, 32 GB RAM and 16 CPU cores. Such pods will be destroyed in 6 hours, unless you request an exception for your namespace (in case you run jupyterhub or some other application controlling the pods for you).

- If you need to run a larger and longer computation, you need to use one of available Workload Controllers. We recommend running those as Jobs - this will closely watch your workload and make sure it ran to completion (exited with 0 status), shut it down to free up the resources, and restart if node was rebooted or something else has happened. Please see the guide on using those. You can use Guaranteed QoS for those.

- In case you need some pod to run idle for a long time, you can use the Deployment controller. Make sure you set minimal request and proper limits for those to get the Burstable QoS


## Running batch jobs

- Highly recommended to use Jobs for any kind of development and computations. 

- This ensures you never lose your work, get the results in the most convenient way, and don't waste resources, since this method does not require any babysitting of processes from you. 

- Once your development is done, you are immediately ready to run a large-scale stuff with no changes to the code and minimal changes in the definition, plus your changes are saved in Git.

- Since jobs in Nautilus can run forever, you can only run jobs with **meaningful `command`** field.

- When job is finished, your pod will stay in Completed state, and Job will have COMPLETIONS field 1/1. For long jobs, the pods can have Error, Evicted, and other states until they finish properly or backoffLimit is exhausted.



### Available GPUs and choosing GPU type
- NVIDIA A10
- NVIDIA A100-SXM4-80GB (consumes much more memory space compared to other GPUs)
- NVIDIA A100-PCIE-40GB
- Quadro RTX 6000
- NVIDIA-GeForce-GTX-1080-Ti
- NVIDIA-GeForce-RTX-2080-Ti
- NVIDIA-GeForce-RTX-3090
- NVIDIA A100 80GB PCIe MIG 1g.10gb
- NVIDIA TITAN-RTX


## Kubernetes commands
Below are some commonly used Kubernetes commands along with brief descriptions:

##### 1. Get Information about Pods
```
kubectl get pods
```

##### 2. Fetch information about the cluster nodes, specifically looking for nodes with NVIDIA GPUs and their corresponding GPU models.
```
kubectl get nodes -L nvidia.com/gpu.product
```

##### 3. Create a new Kubernetes job based on the configuration provided. Jobs are used for running batch processes or scheduled tasks.
```
kubectl create -f create_job_config.yaml
```

##### 4. Execute a command interactively within a specific pod
```
kubectl exec -it <pod name> -- /bin/bash
```

##### 5. Delete a specific pod
```
kubectl delete pod <pod name>
```

##### 5. Deletes a specific job which terminates its associated pods and frees up any allocated resources
```
kubectl delete job <job name>
```

##### 6. Retrieve information about Persistent Volume Claims (PVCs) in the current namespace. Displays details such as PVC names, statuses, and storage capacity. PVCs are used to request persistent storage in a Kubernetes cluster.
```
kubectl get pvc
```
