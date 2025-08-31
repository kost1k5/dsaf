import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './StrategyManager.css';

// --- TYPE DEFINITIONS ---
type StrategyParams = { [key: string]: number | string };

type StrategyInfo = {
  active: boolean;
  params: StrategyParams;
  type: 'signal' | 'grid';
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
    const strategyInfo = strategies[strategyName];
    const isStarting = !strategyInfo.active;

    // Optimistically update UI
    setStrategies(prev => ({
        ...prev,
        [strategyName]: { ...strategyInfo, active: isStarting }
    }));

    try {
        if (strategyInfo.type === 'grid') {
            const endpoint = isStarting ? '/api/grid-bot/start' : '/api/grid-bot/stop';
            // For grid bot, we might need to specify the mode, e.g., 'demo'
            const payload = isStarting ? { mode: 'demo' } : {};
            await axios.post(endpoint, payload);
        } else {
            await axios.post('/api/strategies/status', { statuses: { [strategyName]: isStarting } });
        }
        showFeedback('success', `${toTitleCase(strategyName)} status updated.`, `${strategyName}-toggle`);
        // Refresh status from server to get the real state
        fetchStrategyStatus();
    } catch (error: any) {
        console.error("Failed to update strategy status", error);
        showFeedback('error', error.response?.data?.detail || `Failed to update ${toTitleCase(strategyName)}.`, `${strategyName}-toggle`);
        setStrategies(originalStrategies); // Revert on error
    }
  };

  const handleParamChange = (strategyName: string, paramName: string, value: string) => {
    const isNumeric = !isNaN(Number(value)) && value !== '';
    const strategyInfo = strategies[strategyName];

    // For grid bot, handle symbol as text
    const paramType = typeof strategyInfo.params[paramName];
    const parsedValue = paramType === 'string' ? value : (isNumeric ? parseFloat(value) : value);

    setStrategies(prev => ({
        ...prev,
        [strategyName]: {
            ...prev[strategyName],
            params: {
                ...prev[strategyName].params,
                [paramName]: parsedValue,
            }
        }
    }));
  };

  const handleSaveParams = async (strategyName: string) => {
    const strategyConfig = strategies[strategyName];
    try {
        if (strategyConfig.type === 'grid') {
            await axios.post('/api/grid-bot/settings', strategyConfig.params);
        } else {
            await axios.post('/api/strategies/params', { name: strategyName, params: strategyConfig.params });
        }
        showFeedback('success', `Parameters for ${toTitleCase(strategyName)} saved!`, `${strategyName}-save`);
    } catch (error: any) {
        console.error("Failed to save params", error);
        showFeedback('error', error.response?.data?.detail || `Failed to save parameters.`, `${strategyName}-save`);
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
          {Object.entries(strategies).map(([name, info]) => {
            if (info.type === 'grid') {
              // --- Grid Bot Card ---
              return (
                <div key={name} className={`strategy-card ${info.active ? 'active' : ''}`}>
                  <div className="strategy-card-header">
                    <h3>{toTitleCase(name)} Bot</h3>
                    <span className={`status-pill status-${info.active ? 'active' : 'stopped'}`}>
                      {info.active ? 'Active' : 'Stopped'}
                    </span>
                  </div>
                  <div className="strategy-card-body">
                    <div className="params-form">
                      {Object.entries(info.params).map(([param, value]) => (
                        <div className="param-input-group" key={param}>
                          <label>{toTitleCase(param)}</label>
                          <input
                            type={typeof value === 'string' ? 'text' : 'number'}
                            value={value}
                            onChange={(e) => handleParamChange(name, param, e.target.value)}
                            step="any"
                          />
                        </div>
                      ))}
                    </div>
                    <div className="button-group">
                      <button className="save-button" onClick={() => handleSaveParams(name)}>
                        Save Settings
                      </button>
                       <button
                        className="start-button"
                        onClick={() => handleToggle(name)}
                        disabled={info.active}
                      >
                        Start
                      </button>
                      <button
                        className="stop-button"
                        onClick={() => handleToggle(name)}
                        disabled={!info.active}
                      >
                        Stop
                      </button>
                    </div>
                    {feedback.id === `${name}-save` && <p className={`inline-feedback ${feedback.type}`}>{feedback.message}</p>}
                     {feedback.id === `${name}-toggle` && <p className={`inline-feedback ${feedback.type}`}>{feedback.message}</p>}
                  </div>
                </div>
              );
            } else {
              // --- Signal Strategy Card ---
              return (
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
              );
            }
          })}
        </div>
      )}
    </div>
  );
};

export default StrategyManager;
