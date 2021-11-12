Code to run the experiment on the web server.
Run the whole pipeline by calling [run.sh](run.sh). If this is the first run of `run.sh`, initialize the setup by executing the [init script for the SSL encryption](web-data/init-letsencrypt.sh).

The folder [web-data/nginx](web-data/nginx) contains the configuration file for the HTTP web server, i.e. nginx.

Before you run any of the scripts here, you need to insert your e-email address as well as the domain you want to use 
to host the experiment in some configuration files (e.g., search and replace `yourdomain.tld` and `your@email.tld`).
