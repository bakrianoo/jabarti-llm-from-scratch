"""
tracking.py -- recording what happened during a run
"""

from dataclasses import asdict

import trackio


class Tracker:
    """Logs metrics to Trackio, or to nothing at all.
    """

    def __init__(self, training_config, model_config=None):
        self.run_name = training_config.run_name
        self.project = training_config.project
        self.run_id = None

        run_config = {
            "training": asdict(training_config)
        }

        if model_config is not None:
            run_config["model"] = asdict(model_config)

        run = trackio.init(
            project=self.project,
            name=self.run_name,
            config=run_config
        )

        self.run_id = getattr(run, "id", None)

        print(
            f"trackio: run '{self.run_name}' (id {self.run_id}) "
            f"in project '{self.project}'"
        )

        print(f"  open the dashboard:  trackio show --project {self.project}")

    def log(self, metrics, step):
        trackio.log(metrics, step=step)

    def log_table(self, name, columns, rows, step):

        table_data = [ list(row) for row in rows ]
        table = trackio.Table(
            columns=list(columns),
            data=table_data
        )

        trackio.log({
            name: table
        }, step=step)

    def finish(self):
        trackio.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.finish()
        return False
    
