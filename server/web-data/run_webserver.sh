#!/usr/bin/env bash

# run html server with ssl encryption using docker compose
echo "Spawning nginx+certbot containers"
# make sure nginx log file is created before we mount it
sudo touch /var/log/nginx/access.log
docker-compose -f webserver-docker-compose.yml up -d
sleep 2

nginx_container_id=$(docker-compose -f webserver-docker-compose.yml ps -q nginx)
certbot_container_id=$(docker-compose -f webserver-docker-compose.yml ps -q certbot)
