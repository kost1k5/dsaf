import importlib

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports and returns a strategy class from the src.strategies module.
    """
    try:
        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        # Формируем имя модуля на основе имени стратегии.
        # Например, 'simple_ma' -> 'src.strategies.simple_ma'
        module_name = f"src.strategies.{strategy_name}"

        # Преобразуем имя стратегии из snake_case в PascalCase для получения имени класса.
        # Например, 'simple_ma' -> 'SimpleMaStrategy'
        class_name = "".join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        strategy_module = importlib.import_module(module_name)
        return getattr(strategy_module, class_name)
    
    except (ImportError, AttributeError) as e:
        print(f"Original error loading strategy '{strategy_name}': {e}")
        # Теперь эти переменные определены и могут быть использованы в сообщении об ошибке
        raise ValueError(f"Could not find strategy '{strategy_name}'. "
                         f"Ensure module '{module_name}.py' and class '{class_name}' exist.") from e

