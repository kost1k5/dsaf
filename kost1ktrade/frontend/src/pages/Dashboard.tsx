import { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';
import MasterBotControls from '../components/MasterBotControls';

interface Balances {
  [key: string]: {
    free: number;
    used: number;
    total: number;
  };
}

const Dashboard = () => {
  const [balances, setBalances] = useState<Balances | null>(null);
  const [gridBotStatus, setGridBotStatus] = useState('loading...');
  const [error, setError] = useState('');

  // Centralized status fetching in the Dashboard
  useEffect(() => {
    const fetchAllData = async () => {
      try {
        const [masterStatusRes, signalStatusRes, gridStatusRes] = await axios.all([
          axios.get('/api/master-bot/status'),
          axios.get('/api/signal-bot/status'),
          axios.get('/api/grid-bot/status')
        ]);

        const isMasterRunning = masterStatusRes.data?.master_mode === 'running';
        const isSignalRunning = signalStatusRes.data?.mode !== 'stopped';
        const isGridRunning = gridStatusRes.data?.current_mode !== 'stopped';

        setGridBotStatus(gridStatusRes.data?.current_mode || 'stopped');

        // Fetch balance if any bot is active
        if (isMasterRunning || isSignalRunning || isGridRunning) {
          const balanceRes = await axios.get('/api/balance');
          if (balanceRes.data) {
            setBalances(balanceRes.data);
          } else {
            console.error("Invalid balance response structure:", balanceRes.data);
            setBalances(null); // Clear balance on invalid response
          }
        } else {
          setBalances(null); // No bots running, so no balance to show
        }
        setError(''); // Clear previous errors on successful fetch
      } catch (err: any) {
        console.error("Error fetching dashboard data:", err);
        setError(err.response?.data?.detail || 'Failed to fetch data from server.');
        setBalances(null); // Clear balance on error
      }
    };

    fetchAllData();
    const interval = setInterval(fetchAllData, 5000); // Poll every 5 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  const renderBalances = () => {
      if (!balances) {
          return <p>Start a bot to view balances.</p>;
      }
      // Filter out currencies with zero total balance
      const nonZeroBalances = Object.entries(balances).filter(([, details]) => details.total > 0);

      if (nonZeroBalances.length === 0) {
          return <p>No balances to display.</p>;
      }

      return (
          <ul>
              {nonZeroBalances.map(([currency, details]) => (
                  <li key={currency}>
                      <span className="currency">{currency}</span>
                      <span className="amount">{details.total.toFixed(6)}</span>
                  </li>
              ))}
          </ul>
      );
  }

  return (
    <div className="dashboard">
      <h1>Command Bridge</h1>
      {error && <p className="error-message">{error}</p>}

      <div className="bot-controls-container">
        <MasterBotControls />

        <div className="status-card">
          <h3>Grid Bot</h3>
          <p className={`status-pill status-${gridBotStatus}`}>{gridBotStatus}</p>
          {/* Note: Grid Bot controls are not part of the current scope. */}
          {/* This is just a status display. */}
        </div>
      </div>


      <div className="balance-widget">
        <h2>Asset Balances</h2>
        {renderBalances()}
      </div>
    </div>
  );
};

export default Dashboard;
