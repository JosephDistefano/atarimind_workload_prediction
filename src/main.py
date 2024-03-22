from pathlib import Path
import yaml
from utils import skip_run
from features.epoch_data_per_second import epoch_data_into_seconds
from features.ccl_features import calculate_CCL_features
from features.eye_features import calculate_eye_features
from features.fusion_features import calculate_fusion_features
from visualization.visualize_epochs import visualize_epochs
from classification.classification_eye_gaze import classification_eye_gaze
from classification.feature_matrices import create_feature_matrix_eye_ccl

config_path = Path(__file__).parents[1] / "src/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "calculate all fusion features based on CCL and eye gaze") as check, check():
    epoch_data_into_seconds(config)

with skip_run("skip", "calculate CCL features") as check, check():
    calculate_CCL_features(config)

with skip_run("skip", "calculate gaze features") as check, check():
    calculate_eye_features(config)

with skip_run("run", "calculate fusion features") as check, check():
    calculate_fusion_features(config)

with skip_run("skip", "visualize epochs and features") as check, check():
    visualize_epochs(config)

with skip_run("skip", "create feature matrix with gaze and CCL features") as check, check():
    create_feature_matrix_eye_ccl(config)

with skip_run("skip", "use eye features and CCL features to classify workload (no fusion features"):
    classification_eye_gaze(config)


