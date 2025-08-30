import importlib

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports and returns a strategy class from the src.strategies module.
    """
    try:
        # e.g., 'macd' -> 'src.strategies.macd'
        module_name = f"src.strategies.{strategy_name.lower()}"
        # e.g., 'macd' -> 'MacdStrategy', 'sma_crossover' -> 'SmaCrossoverStrategy'
        class_name = f"{strategy_name.replace('_', ' ').title().replace(' ', '')}Strategy"

        strategy_module = importlib.import_module(module_name)
        return getattr(strategy_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find strategy '{strategy_name}'. "
                         f"Ensure module '{module_name}.py' and class '{class_name}' exist.") from e
