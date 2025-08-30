import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './MasterBotControls.css';

interface MasterStatus {
  master_mode: string;
  market_state?: string;
  adx_value?: number;
}

interface SignalStatus {
    mode: string;
    strategy_name: string | null;
}

const MasterBotControls = () => {
    const [masterStatus, setMasterStatus] = useState<MasterStatus>({ master_mode: 'loading...' });
    const [signalStatus, setSignalStatus] = useState<SignalStatus>({ mode: 'loading...', strategy_name: null });
    const [isLoading, setIsLoading] = useState(false);
    const [feedback, setFeedback] = useState({ type: '', message: '' });

    const fetchStatus = async () => {
        try {
            const [masterRes, signalRes] = await axios.all([
                axios.get('/api/master-bot/status'),
                axios.get('/api/signal-bot/status')
            ]);
            setMasterStatus(masterRes.data);
            setSignalStatus(signalRes.data);
        } catch (error) {
            console.error("Failed to fetch bot statuses", error);
            setMasterStatus({ master_mode: 'error' });
            setSignalStatus({ mode: 'error', strategy_name: 'Unknown' });
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000); // Poll every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const handleStart = async () => {
        setIsLoading(true);
        setFeedback({ type: '', message: '' });
        try {
            const res = await axios.post('/api/master-bot/start');
            setFeedback({ type: 'success', message: res.data.message });
            fetchStatus(); // Refresh status immediately
        } catch (err: any) {
            setFeedback({ type: 'error', message: err.response?.data?.detail || 'Failed to start master bot.' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleStop = async () => {
        setIsLoading(true);
        setFeedback({ type: '', message: '' });
        try {
            const res = await axios.post('/api/master-bot/stop');
            setFeedback({ type: 'success', message: res.data.message });
            fetchStatus(); // Refresh status immediately
        } catch (err: any) {
            setFeedback({ type: 'error', message: err.response?.data?.detail || 'Failed to stop master bot.' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="master-bot-controls">
            <h3>Autonomous Mode</h3>
            <div className="status-container">
                <div className="status-item">
                    <span>Master Controller:</span>
                    <span className={`status-pill status-${masterStatus.master_mode}`}>{masterStatus.master_mode}</span>
                </div>
                {masterStatus.master_mode === 'running' && (
                    <>
                        <div className="status-item">
                            <span>Market State:</span>
                            <span className="status-value">{masterStatus.market_state || 'N/A'} (ADX: {masterStatus.adx_value || 'N/A'})</span>
                        </div>
                        <div className="status-item">
                            <span>Active Strategy:</span>
                            <span className="status-value">{signalStatus.strategy_name || 'None'}</span>
                        </div>
                    </>
                )}
            </div>
            <div className="button-group">
                <button onClick={handleStart} disabled={isLoading || masterStatus.master_mode !== 'stopped'}>
                    ▶ Start Autonomous Mode
                </button>
                <button onClick={handleStop} disabled={isLoading || masterStatus.master_mode === 'stopped'} className="stop-button">
                    ■ Stop Autonomous Mode
                </button>
            </div>
            {feedback.message && (
                <div className={`feedback-message ${feedback.type}`}>
                    {feedback.message}
                </div>
            )}
        </div>
    );
};

export default MasterBotControls;
