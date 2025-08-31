import importlib
import numpy as np

# Monkey-patch numpy to fix pandas-ta import issue.
# Some versions of pandas-ta (which is not pinned) use the deprecated np.NaN
# attribute instead of np.nan. This causes an ImportError on newer numpy versions.
# Pinning all dependencies to a compatible state is complex, so this
# patch ensures the bot can run even if pandas-ta is updated to a version
# with this issue.
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports and returns a strategy class from the src.strategies module.
    """
    module_name = f"src.strategies.{strategy_name}"
    class_name = "".join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"

    try:
        strategy_module = importlib.import_module(module_name)
        return getattr(strategy_module, class_name)
    
    except (ImportError, AttributeError) as e:
        print(f"Original error loading strategy '{strategy_name}': {e}")
        raise ValueError(f"Could not find strategy '{strategy_name}'. "
                         f"Ensure module '{module_name}.py' and class '{class_name}' exist.") from e
