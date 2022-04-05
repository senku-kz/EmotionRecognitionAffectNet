# EmotionRecognition

## Order of execution
python db_insert_records_into_sql.py

python db_add_head_position_information.py

python models_train.py

python models_test.py

python models_plot.py

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

docker run --rm --gpus device=0 --name emotion -d -v ${PWD}:/app emotion_recognition_env


docker exec -it emotion bash


docker build -t create_sqlite:latest -f dockerfile_createSQLite .
docker run --name sqlite -d -v ${PWD}:/app create_sqlite



docker save emotion_recognition_env > emotion_recognition_env.tar
docker load < emotion_recognition_env.tar


docker save emotion_recognition_env:gpu > emotion_recognition_env_gpu.tar
docker load < emotion_recognition_env_gpu.tar

#certutil -hashfile <file> MD5

certutil -hashfile emotion_recognition_env.tar MD5

md5sum emotion_recognition_env.tar


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