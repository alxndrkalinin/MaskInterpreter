"""
Initialize environment variables with default values if not already set.
This module ensures all required environment variables are available throughout the codebase.
"""
import os


def init_env_vars():
    """Initialize environment variables with default values if not already set."""
    
    # Set default paths if environment variables are not defined
    if 'DATA_MODELS_PATH' not in os.environ:
        os.environ['DATA_MODELS_PATH'] = '/mnt/new_groups/assafza_group/assafza'
    if 'DATA_PATH' not in os.environ:
        os.environ['DATA_PATH'] = '/mnt/new_groups/assafza_group/assafza/full_cells_fovs/train_test_list'
        
    if 'MODELS_PATH' not in os.environ:
            os.environ['MODELS_PATH'] = '/mnt/new_groups/assafza_group/assafza/lion_models_clean/models'
    
    if 'REPO_LOCAL_PATH' not in os.environ:
        os.environ['REPO_LOCAL_PATH'] = '/home/lionb'
    
    if 'EXAMPLE_DATA_PATH' not in os.environ:
        os.environ['EXAMPLE_DATA_PATH'] = '/mnt/new_groups/assafza_group/assafza/lion_models_clean/example_data/'


# Auto-initialize when module is imported
init_env_vars()
