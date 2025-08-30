import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './StrategyManager.css';

type StrategyParams = { [key: string]: number | string };
type AvailableStrategies = { [key: string]: StrategyParams };

const StrategyManager = () => {
  const [availableStrategies, setAvailableStrategies] = useState<AvailableStrategies>({});
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [params, setParams] = useState<StrategyParams>({});
  const [mode, setMode] = useState('demo');
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState({ type: '', message: '' });

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await axios.get('/api/strategies');
        setAvailableStrategies(response.data);
        // Set the first strategy as the default selection
        const firstStrategyName = Object.keys(response.data)[0];
        if (firstStrategyName) {
          setSelectedStrategy(firstStrategyName);
          setParams(response.data[firstStrategyName]);
        }
      } catch (error) {
        console.error("Failed to fetch strategies", error);
        setFeedback({ type: 'error', message: 'Could not load strategies from server.' });
      }
    };
    fetchStrategies();
  }, []);

  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newStrategyName = e.target.value;
    setSelectedStrategy(newStrategyName);
    setParams(availableStrategies[newStrategyName] || {});
  };

  const handleParamChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    // Handle both number and potential string inputs gracefully
    const isNumeric = /^\d*\.?\d*$/.test(value) && value !== '';
    setParams({
      ...params,
      [name]: isNumeric ? parseFloat(value) : value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setFeedback({ type: '', message: '' });

    const requestBody = {
      mode,
      strategy_name: selectedStrategy,
      strategy_params: params,
    };

    try {
      const response = await axios.post('/api/signal-bot/start', requestBody);
      setFeedback({ type: 'success', message: response.data.message || 'Bot started successfully!' });
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.response?.data?.detail || 'Failed to start bot.' });
    } finally {
      setIsLoading(false);
    }
  };

  const renderStrategyParams = () => {
    if (!selectedStrategy) {
      return <p>Select a strategy to configure its parameters.</p>;
    }
    return Object.entries(params).map(([paramName, paramValue]) => (
      <div className="form-group" key={paramName}>
        <label htmlFor={paramName}>{paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</label>
        <input
          type="number"
          id={paramName}
          name={paramName}
          value={paramValue}
          onChange={handleParamChange}
          step="any"
        />
      </div>
    ));
  };

  return (
    <div className="strategy-manager">
      <h2>Strategy Manager</h2>
      <p>Configure and launch a new signal bot.</p>

      <form onSubmit={handleSubmit} className="strategy-form">
        <div className="form-group">
          <label htmlFor="strategy-select">Strategy</label>
          <select
            id="strategy-select"
            value={selectedStrategy}
            onChange={handleStrategyChange}
            disabled={!Object.keys(availableStrategies).length}
          >
            <option value="" disabled>-- Select a Strategy --</option>
            {Object.keys(availableStrategies).map(name => (
              <option key={name} value={name}>{name.replace(/_/g, ' ').toUpperCase()}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
            <label htmlFor="mode-select">Mode</label>
            <select id="mode-select" value={mode} onChange={e => setMode(e.target.value)}>
                <option value="demo">Demo</option>
                <option value="real">Real</option>
            </select>
        </div>

        <fieldset>
          <legend>Strategy Parameters</legend>
          {renderStrategyParams()}
        </fieldset>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Launching...' : 'Launch Bot'}
        </button>
      </form>

      {feedback.message && (
        <div className={`feedback-message ${feedback.type}`}>
          {feedback.message}
        </div>
      )}
    </div>
  );
};

export default StrategyManager;
