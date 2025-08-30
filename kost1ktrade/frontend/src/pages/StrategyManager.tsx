import React, { useState } from 'react';
import axios from 'axios';
import './StrategyManager.css';

// A simple type for the form state
type StrategyParams = {
  [key: string]: number | string;
};

const StrategyManager = () => {
  const [selectedStrategy, setSelectedStrategy] = useState('MACD');
  const [params, setParams] = useState<StrategyParams>({
    fast_period: 12,
    slow_period: 26,
    signal_period: 9,
  });
  const [mode, setMode] = useState('demo');
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState({ type: '', message: '' });

  const handleParamChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams({
      ...params,
      [e.target.name]: parseInt(e.target.value, 10),
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

  // For now, the form is hardcoded for MACD.
  // In the future, this would be dynamically generated based on the selected strategy.
  const renderStrategyParams = () => {
    switch (selectedStrategy) {
      case 'MACD':
        return (
          <>
            <div className="form-group">
              <label htmlFor="fast_period">Fast Period</label>
              <input
                type="number"
                id="fast_period"
                name="fast_period"
                value={params.fast_period || ''}
                onChange={handleParamChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="slow_period">Slow Period</label>
              <input
                type="number"
                id="slow_period"
                name="slow_period"
                value={params.slow_period || ''}
                onChange={handleParamChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="signal_period">Signal Period</label>
              <input
                type="number"
                id="signal_period"
                name="signal_period"
                value={params.signal_period || ''}
                onChange={handleParamChange}
              />
            </div>
          </>
        );
      // Add cases for other strategies like 'RSI', 'SMA_Crossover' here
      default:
        return <p>Select a strategy to configure its parameters.</p>;
    }
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
            onChange={e => setSelectedStrategy(e.target.value)}
          >
            <option value="MACD">MACD</option>
            {/* Future options: <option value="RSI">RSI</option> */}
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
