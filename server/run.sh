#!/usr/bin/env bash

CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
CROSS_MARK="\033[0;31m\xE2\x9D\x8C\033[0m"

nginx_container_id=""

ask_to_exit() {
    echo ""
    while true; do
        read -p "Stopping runtime... Do you wish to stop the still running containers? [(y)es, (n)o, (c)ancel]: " ync
        case $ync in
            [Yy]* ) stop_containers; exit;;
            [Nn]* ) exit;;
            [Cc]* ) break;;
            * ) echo "Please answer (y)es, (n)o or (c)ancel.";;
        esac
    done
}

stop_containers() {
    if ! [[ -z "$nginx_container_id" ]]; then
        echo "Stopping nginx + certbot + bouncer containers..."
        docker stop $nginx_container_id > /dev/null 2>&1
        nginx_container_running=$(docker inspect -f '{{.State.Running}}' $nginx_container_id)
        if [[ "$nginx_container_running" = "false" ]]; then
            echo -e "\t$CHECK_MARK Stopped ginx + certbot + bouncer containers"
        else
            echo -e "\t$CROSS_MARK Couldn't stop nginx + certbot + bouncer containers"
        fi
    fi
    docker-compose -f ./web-data/docker-compose.yml down
}

trap ask_to_exit INT

# run html server with ssl encryption using docker compose
echo "Spawning nginx + certbot + bouncer containers"
# make sure nginx log file is created before we mount it
sudo mkdir -p /var/log/nginx/
sudo mkdir -p /var/log/bouncer/
sudo touch /var/log/nginx/access.log
sudo touch /var/log/bouncer/log.log
docker-compose -f ./web-data/docker-compose.yml up -d
sleep 5

nginx_container_id=$(docker-compose -f ./web-data/docker-compose.yml ps -q nginx)
certbot_container_id=$(docker-compose -f ./web-data/docker-compose.yml ps -q certbot)
bouncer_container_id=$(docker-compose -f ./web-data/docker-compose.yml ps -q bouncer)

echo "All containers running..."

while true; do
    nginx_container_running=$(docker inspect -f '{{.State.Running}}' $nginx_container_id)
    certbot_container_running=$(docker inspect -f '{{.State.Running}}' $certbot_container_id)
    bouncer_container_running=$(docker inspect -f '{{.State.Running}}' $bouncer_container_id)

    if [[ "$nginx_container_running" = "false" ]]; then
        echo -e "\t$CROSS_MARK Nginx container stopped"
        ask_to_exit
    fi

    if [[ "$certbot_container_running" = "false" ]]; then
        echo -e "\t$CROSS_MARK Certbot/letsencrypt container stopped"
        ask_to_exit
    fi

    if [[ "bouncer_container_running" = "false" ]]; then
        echo -e "\t$CROSS_MARK Bouncer container stopped"
        ask_to_exit
    fi

    sleep .5
done