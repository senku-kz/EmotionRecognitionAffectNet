# EmotionRecognition

## Order of execution
nohup python path/to/script.py &

nohup python mat2csv.py &

nohup python csv2sqlite.py &

nohup python move_files_by_label.py &

nohup python train_model.py &


## for venv
pip freeze > requirements.txt

pip install -r requirements.txt



## for docker
sudo systemctl restart docker
docker pull python:3.9.0
docker run -it python:3.9.0


cd docker/
# docker build .
docker build -t emotion_recognition_env:latest .

# docker run --rm --name emotion -v ${PWD}:/app emotion_recognition_env
# docker run --rm --name emotion -ti -v ${PWD}:/app emotion_recognition_env bash
# docker run --rm --name emotion -d -v ${PWD}:/app emotion_recognition_env tail -f /dev/null
# docker run --rm --name emotion -ti -v ${PWD}:/app emotion_recognition_env
docker run --rm --name emotion -d -v ${PWD}:/app emotion_recognition_env


docker exec -it emotion bash


docker save emotion_recognition_env > emotion_recognition_env.tar
docker load < emotion_recognition_env.tar


https://devopscube.com/keep-docker-container-running/
https://medium.com/bb-tutorials-and-thoughts/docker-export-vs-save-7053504546e5

# ====
You can use docker commit:

Start your container sudo docker run IMAGE_NAME
Access your container using bash: sudo docker exec -it CONTAINER_ID bash
Install whatever you need inside the container
Exit container's bash
Commit your changes: sudo docker commit CONTAINER_ID NEW_IMAGE_NAME

If you run now docker images, you will see NEW_IMAGE_NAME listed under your local images.

Next time, when starting the docker container, use the new docker image you just created:
sudo docker run **NEW_IMAGE_NAME** - this one will include your additional installations.

Answer based on the following tutorial: How to commit changes to docker image
# ===