import { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';
import MasterBotControls from '../components/MasterBotControls';

interface Balances {
  [key: string]: number;
}

const Dashboard = () => {
  const [gridBotStatus, setGridBotStatus] = useState('loading...');
  const [balances, setBalances] = useState<Balances | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    // This effect now only fetches grid status and balance.
    // Signal bot and master bot status are handled within their own components.
    const fetchStatus = async () => {
      try {
        const gridStatusRes = await axios.get('/api/grid-bot/status');
        const gridStatus = gridStatusRes.data.current_mode;
        setGridBotStatus(gridStatus);

        // Fetch balance if grid bot is active.
        // The master bot component will also trigger balance fetching if it's active.
        if (gridStatus !== 'stopped') {
          const balanceRes = await axios.get('/api/balance');
          // The balance response is a complex object. We want the 'total' part.
          if (balanceRes.data && balanceRes.data.total) {
            setBalances(balanceRes.data.total);
          } else {
            console.error("Invalid balance response structure:", balanceRes.data);
            setBalances(null);
          }
        } else {
          setBalances(null);
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

      <MasterBotControls />

      <div className="other-bots-section">
        <div className="status-grid">
            <div className="status-card">
            <h3>Grid Bot</h3>
            <p className={`status-pill status-${gridBotStatus}`}>{gridBotStatus}</p>
            </div>
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
    </div>
  );
};

export default Dashboard;
