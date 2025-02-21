from .models import actdyn_mclean2003, actdyn_mclean2003_improved, actdyn_degroote2016_original, actdyn_degroote2016

# Dictionary for easy lookup by model name
MODEL_FUNCTIONS = {
    'McLean2003': actdyn_mclean2003,
    'McLean2003Improved': actdyn_mclean2003_improved,
    'DeGroote2016Original': actdyn_degroote2016_original,
    'DeGroote2016': actdyn_degroote2016
}