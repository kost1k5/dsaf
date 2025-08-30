import { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';
import BotControls from '../components/BotControls';

interface Balances {
  [key: string]: number;
}

const Dashboard = () => {
  const [signalBotStatus, setSignalBotStatus] = useState('loading...');
  const [gridBotStatus, setGridBotStatus] = useState('loading...');
  const [balances, setBalances] = useState<Balances | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [signalStatus, gridStatus] = await axios.all([
          axios.get('/api/signal-bot/status'),
          axios.get('/api/grid-bot/status')
        ]);
        setSignalBotStatus(signalStatus.data.current_mode);
        setGridBotStatus(gridStatus.data.current_mode);

        // Try to fetch balance only if a bot is active
        if (signalStatus.data.current_mode !== 'stopped' || gridStatus.data.current_mode !== 'stopped') {
          const balanceRes = await axios.get('/api/balance');
          setBalances(balanceRes.data);
        } else {
          setBalances(null); // No bot active, no balance to show
        }

      } catch (err: any) {
        console.error("Error fetching status or balance:", err);
        setError(err.response?.data?.detail || 'Failed to fetch data from server.');
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Poll every 5 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return (
    <div className="dashboard">
      <h1>Command Bridge</h1>
      {error && <p className="error-message">{error}</p>}

      <div className="status-grid">
        <div className="status-card">
          <h3>Signal Bot</h3>
          <p className={`status-pill status-${signalBotStatus}`}>{signalBotStatus}</p>
        </div>
        <div className="status-card">
          <h3>Grid Bot</h3>
          <p className={`status-pill status-${gridBotStatus}`}>{gridBotStatus}</p>
        </div>
      </div>

      <div className="balance-widget">
        <h2>Asset Balances</h2>
        {balances ? (
          <ul>
            {Object.entries(balances).map(([currency, amount]) => (
              <li key={currency}>
                <span className="currency">{currency}</span>
                <span className="amount">{amount.toFixed(6)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p>Start a bot to view balances.</p>
        )}
      </div>

      <BotControls />
    </div>
  );
};

export default Dashboard;
