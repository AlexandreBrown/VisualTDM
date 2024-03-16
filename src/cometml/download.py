from pathlib import Path
from comet_ml import Experiment
from comet_ml import API


def download_artifact(api_key: str, project_name: str, workspace: str, artifact_name: str, output_dir: Path) -> Path:
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)
    artifact = experiment.get_artifact(artifact_name)
    artifact_path = Path(artifact.assets[0].logical_path)
    artifact.download(output_dir)
    
    # Workaround for deleting the dummy experiment created during the download
    api = API(api_key=api_key)
    api.delete_experiment(experiment_key=experiment.get_key())
    
    return output_dir / artifact_path
