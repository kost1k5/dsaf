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
  const [feedback, setFeedback] = useState({ type: '', message: '' });

  // Fetch strategies and their statuses on component mount
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

  // Handler for toggling a strategy's active status
  const handleToggle = async (strategyName: string) => {
    const originalStrategies = { ...strategies };
    const updatedStrategy = {
      ...strategies[strategyName],
      active: !strategies[strategyName].active,
    };
    const updatedStrategies = {
      ...strategies,
      [strategyName]: updatedStrategy,
    };

    // Optimistically update the UI
    setStrategies(updatedStrategies);

    // Create the request body for the single strategy update
    const requestBody = {
        statuses: {
            [strategyName]: updatedStrategy.active
        }
    };

    try {
      await axios.post('/api/strategies/status', requestBody);
      // Success, the optimistic update was correct.
    } catch (error) {
      console.error("Failed to update strategy status", error);
      setFeedback({ type: 'error', message: `Failed to update ${toTitleCase(strategyName)}.` });
      // Revert the UI on failure
      setStrategies(originalStrategies);
    }
  };

  return (
    <div className="strategy-manager-page">
      <h2>Strategy Manager</h2>
      <p>Activate or deactivate strategies for the Master Bot to use.</p>

      {isLoading ? (
        <p>Loading strategies...</p>
      ) : (
        <div className="strategy-list-container">
          {Object.entries(strategies).map(([name, info]) => (
            <div key={name} className={`strategy-card ${info.active ? 'active' : ''}`}>
              <div className="strategy-card-header">
                <h3>{toTitleCase(name)}</h3>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={info.active}
                    onChange={() => handleToggle(name)}
                  />
                  <span className="slider round"></span>
                </label>
              </div>
              <div className="strategy-card-body">
                <p>Default Parameters:</p>
                <ul>
                  {Object.entries(info.params).map(([param, value]) => (
                    <li key={param}>
                      <strong>{toTitleCase(param)}:</strong> {value}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      )}

      {feedback.message && (
        <div className={`feedback-message ${feedback.type}`}>
          {feedback.message}
        </div>
      )}
    </div>
  );
};

export default StrategyManager;
