Bouncer is a flask app that checks that no MTurk worker participates more than once in an experiment.
Before MTurk workers have the chance to work on a HIT our UI has to check whether the bouncer lets them pass to the experiment.
Once they have finished the experiment, our UI has to send a request back to bouncer to ban them from further participation.
Previous participation will be logged in an SQLLite3 database.

# Good to Know

## Workflow

### For Testing in Sandbox
To restart the service and reset the database (i.e. create a new database), execute
`docker-compose -f ~/causal_visualizations/server/web-data/docker-compose.yml restart bouncer`
and wait (quite a while) until you see `Restarting webdata_bouncer_1 ... done`.


### For Real
Run the experiment.

Finally, make a backup of `database.db` as well as a local copy.

