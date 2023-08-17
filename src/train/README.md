
# Setting up GPU with Docker

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
evelyn@EveningTUF:/mnt/g/Repositories/opal/scripts/train$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
evelyn@EveningTUF:/mnt/g/Repositories/opal/scripts/train$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```