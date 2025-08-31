import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './StrategyManager.css';

// --- TYPE DEFINITIONS ---
type StrategyParams = { [key: string]: number | string };

type StrategyInfo = {
  active: boolean;
  params: StrategyParams;
};

type StrategiesStatus = {
  [strategyName: string]: StrategyInfo;
};

// --- HELPER FUNCTIONS ---
const toTitleCase = (str: string) => {
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};


// --- MAIN COMPONENT ---
const StrategyManager = () => {
  const [strategies, setStrategies] = useState<StrategiesStatus>({});
  const [isLoading, setIsLoading] = useState(true);
  const [feedback, setFeedback] = useState<{type: string, message: string, id?: string}>({ type: '', message: '' });

  const fetchStrategyStatus = useCallback(async () => {
    try {
      const response = await axios.get('/api/strategies/status');
      setStrategies(response.data);
    } catch (error) {
      console.error("Failed to fetch strategies status", error);
      setFeedback({ type: 'error', message: 'Could not load strategies from server.' });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStrategyStatus();
  }, [fetchStrategyStatus]);

  const showFeedback = (type: string, message: string, id?: string) => {
    setFeedback({ type, message, id });
    setTimeout(() => setFeedback({ type: '', message: '' }), 3000);
  };

  const handleToggle = async (strategyName: string) => {
    const originalStrategies = { ...strategies };
    const updatedStrategy = { ...strategies[strategyName], active: !strategies[strategyName].active };
    setStrategies(prev => ({ ...prev, [strategyName]: updatedStrategy }));

    try {
      await axios.post('/api/strategies/status', { statuses: { [strategyName]: updatedStrategy.active } });
      showFeedback('success', `${toTitleCase(strategyName)} status updated.`, `${strategyName}-toggle`);
    } catch (error) {
      console.error("Failed to update strategy status", error);
      showFeedback('error', `Failed to update ${toTitleCase(strategyName)}.`, `${strategyName}-toggle`);
      setStrategies(originalStrategies);
    }
  };

  const handleParamChange = (strategyName: string, paramName: string, value: string) => {
    const isNumeric = !isNaN(Number(value)) && value !== '';
    setStrategies(prev => ({
        ...prev,
        [strategyName]: {
            ...prev[strategyName],
            params: {
                ...prev[strategyName].params,
                [paramName]: isNumeric ? parseFloat(value) : value,
            }
        }
    }));
  };

  const handleSaveParams = async (strategyName: string) => {
    const strategyConfig = strategies[strategyName];
    try {
      await axios.post('/api/strategies/params', { name: strategyName, params: strategyConfig.params });
      showFeedback('success', `Parameters for ${toTitleCase(strategyName)} saved!`, `${strategyName}-save`);
    } catch (error) {
      console.error("Failed to save params", error);
      showFeedback('error', `Failed to save parameters for ${toTitleCase(strategyName)}.`, `${strategyName}-save`);
    }
  };

  return (
    <div className="strategy-manager-page">
      <h2>Strategy Manager</h2>
      <p>Activate strategies for the Master Bot and customize their parameters.</p>

      {isLoading ? (
        <p>Loading strategies...</p>
      ) : (
        <div className="strategy-list-container">
          {Object.entries(strategies).map(([name, info]) => (
            <div key={name} className={`strategy-card ${info.active ? 'active' : ''}`}>
              <div className="strategy-card-header">
                <h3>{toTitleCase(name)}</h3>
                <label className="switch">
                  <input type="checkbox" checked={info.active} onChange={() => handleToggle(name)} />
                  <span className="slider round"></span>
                </label>
              </div>
              <div className="strategy-card-body">
                <div className="params-form">
                  {Object.entries(info.params).map(([param, value]) => (
                    <div className="param-input-group" key={param}>
                      <label>{toTitleCase(param)}</label>
                      <input
                        type="number"
                        value={value}
                        onChange={(e) => handleParamChange(name, param, e.target.value)}
                        step="any"
                      />
                    </div>
                  ))}
                </div>
                <button className="save-button" onClick={() => handleSaveParams(name)}>
                  Save Parameters
                </button>
                {feedback.id === `${name}-save` && <p className={`inline-feedback ${feedback.type}`}>{feedback.message}</p>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default StrategyManager;
