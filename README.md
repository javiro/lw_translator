# kuberay_tests

## Docker deployment

Run the following bash commands to deploy the ML model in a Docker container.

- Build the image: ```docker buildx build --tag 'lw_translator' .```
- Run the image: ```docker run -it -p 8000:8000 'lw_translator'```

## Kubernetes deployment

Run the following bash commands to deploy the ML model in a Kubernetes cluster.

- Install the kuberay-operator: ```helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0```
- Deploy the ML model: ```kubectl apply -f ray-service.translator.yaml```

### Set up the Kubernetes cluster on GCP

- Create a new project
- Go to *Kubernetes engine*
- Create a cluster:
    - ```Name: lw-translator-cluster```
    - ```Region: europe-southwest1```
- Click on ```lw-translator-cluster```
- Connect (running in a shell): ```gcloud container clusters get-credentials lw-translator-cluster --region europe-southwest1 --project lw-translator```
- Get the yaml file: ```curl -LO https://raw.githubusercontent.com/javiro/kuberay_tests/main/ray-service.translator.yaml```
- Install kuberay-operator: ```helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0```
- Port forward: ```kubectl port-forward svc/rayservice-translator-head-svc 8000:8000```
- Deploy the model: ```kubectl apply -f ray-service.translator.yaml```
- Test it in a new shell via ```ipython```: 

```python
import requests
english_text = "Tomorrow will rain."
response = requests.post("http://localhost:8000", json=english_text)
french_text = response.text
print(french_text)
```
- Clean it:

```sh
kubectl delete -f ray-service.translator.yaml
helm uninstall kuberay-operator
```

- Delete the cluster
