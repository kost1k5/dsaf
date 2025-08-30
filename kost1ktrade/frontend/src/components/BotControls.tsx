import React, { useState } from 'react';
import axios from 'axios';
import './BotControls.css';

const BotControls = () => {
  const [gridSymbol, setGridSymbol] = useState('BTC/USDT');
  const [gridLow, setGridLow] = useState('50000');
  const [gridHigh, setGridHigh] = useState('60000');
  const [gridNum, setGridNum] = useState('10');
  const [gridAmount, setGridAmount] = useState('0.001');
  const [message, setMessage] = useState('');

  const handleApiCall = async (url: string, payload?: object) => {
    try {
      const response = await axios.post(url, payload);
      setMessage(response.data.message || 'Success!');
    } catch (err: any) {
      setMessage(err.response?.data?.detail || 'An error occurred.');
    }
    setTimeout(() => setMessage(''), 3000); // Clear message after 3s
  };

  const handleStartGrid = (e: React.FormEvent) => {
    e.preventDefault();
    const payload = {
      mode: 'demo', // Hardcoded to demo for safety
      symbol: gridSymbol,
      amount_per_grid: parseFloat(gridAmount),
      grid_range_low: parseFloat(gridLow),
      grid_range_high: parseFloat(gridHigh),
      num_grids: parseInt(gridNum)
    };
    handleApiCall('/api/grid-bot/start', payload);
  };

  return (
    <div className="controls-container">
      {message && <div className="message-toast">{message}</div>}

      <div className="control-card">
        <h3>Signal Bot</h3>
        <p>Runs a signal-based strategy (e.g., SMA Crossover).</p>
        <div className="button-group">
          <button onClick={() => handleApiCall('/api/signal-bot/start', { mode: 'demo' })}>Start Demo</button>
          <button className="real-button" onClick={() => handleApiCall('/api/signal-bot/start', { mode: 'real' })}>Start Real</button>
          <button onClick={() => handleApiCall('/api/signal-bot/stop')}>Stop</button>
        </div>
      </div>

      <div className="control-card">
        <h3>Grid Bot</h3>
        <form onSubmit={handleStartGrid}>
          <div className="form-grid">
            <input type="text" value={gridSymbol} onChange={e => setGridSymbol(e.target.value)} placeholder="Symbol" />
            <input type="number" value={gridLow} onChange={e => setGridLow(e.target.value)} placeholder="Low Price" />
            <input type="number" value={gridHigh} onChange={e => setGridHigh(e.target.value)} placeholder="High Price" />
            <input type="number" value={gridNum} onChange={e => setGridNum(e.target.value)} placeholder="# of Grids" />
            <input type="number" value={gridAmount} onChange={e => setGridAmount(e.target.value)} placeholder="Amount/Grid" step="0.0001" />
          </div>
          <div className="button-group">
            <button type="submit">Start Demo Grid</button>
            <button type="button" onClick={() => handleApiCall('/api/grid-bot/stop')}>Stop</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default BotControls;
