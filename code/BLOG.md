install crds and kubeflow training operators use this instead of using kustomize and all that. No need for seperate crds and training opertaor directories or files.

kubectl apply -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=master"

kubectl exec --stdin --tty predict-service -- /bin/bash