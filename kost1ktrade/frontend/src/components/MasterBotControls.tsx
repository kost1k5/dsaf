import React, { useState, useEffect, useCallback } from 'react';
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
    const [targetMode, setTargetMode] = useState('demo');
    const [isLoading, setIsLoading] = useState(false);
    const [feedback, setFeedback] = useState({ type: '', message: '' });

    const fetchStatus = useCallback(async () => {
        try {
            const [masterRes, signalRes, settingsRes] = await axios.all([
                axios.get('/api/master-bot/status'),
                axios.get('/api/signal-bot/status'),
                axios.get('/api/master-bot/settings'),
            ]);

            if (masterRes.data && typeof masterRes.data.master_mode !== 'undefined') {
                setMasterStatus(masterRes.data);
            } else {
                console.error("Invalid master bot status response:", masterRes.data);
            }

            if (signalRes.data && typeof signalRes.data.mode !== 'undefined') {
                setSignalStatus(signalRes.data);
            } else {
                console.error("Invalid signal bot status response:", signalRes.data);
            }

            if (settingsRes.data && typeof settingsRes.data.target_mode !== 'undefined') {
                setTargetMode(settingsRes.data.target_mode);
            } else {
                console.error("Invalid master bot settings response:", settingsRes.data);
            }

        } catch (error) {
            console.error("Failed to fetch bot statuses", error);
            setMasterStatus({ master_mode: 'error' });
            setSignalStatus({ mode: 'error', strategy_name: 'Unknown' });
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, [fetchStatus]);

    const showFeedback = (type: string, message: string) => {
        setFeedback({ type, message });
        setTimeout(() => setFeedback({ type: '', message: '' }), 4000);
    };

    const handleModeChange = async (newMode: 'demo' | 'real') => {
        if (newMode === 'real') {
            const confirmation = window.confirm(
                "You are about to switch to REAL TRADING mode.\n\n" +
                "Please confirm that you understand the risks and have configured your API keys correctly.\n\n" +
                "The bot will use REAL aFUNDS."
            );
            if (!confirmation) {
                return; // User cancelled the action
            }
        }

        try {
            await axios.post('/api/master-bot/settings', { target_mode: newMode });
            setTargetMode(newMode);
            showFeedback('success', `Target mode switched to ${newMode.toUpperCase()}.`);
        } catch (err: any) {
            showFeedback('error', err.response?.data?.detail || 'Failed to switch mode.');
        }
    };

    const handleStart = async () => {
        setIsLoading(true);
        try {
            const res = await axios.post('/api/master-bot/start');
            showFeedback('success', res.data.message);
            fetchStatus();
        } catch (err: any) {
            showFeedback('error', err.response?.data?.detail || 'Failed to start master bot.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleStop = async () => {
        setIsLoading(true);
        try {
            const res = await axios.post('/api/master-bot/stop');
            showFeedback('success', res.data.message);
            fetchStatus();
        } catch (err: any) {
            showFeedback('error', err.response?.data?.detail || 'Failed to stop master bot.');
        } finally {
            setIsLoading(false);
        }
    };

    const isBotStopped = masterStatus.master_mode === 'stopped';

    return (
        <div className="master-bot-controls">
            <div className="controls-header">
                <h3>Autonomous Mode</h3>
                <div className={`mode-selector ${!isBotStopped ? 'disabled' : ''}`}>
                    <button
                        className={targetMode === 'demo' ? 'active' : ''}
                        onClick={() => handleModeChange('demo')}
                        disabled={!isBotStopped}
                    >
                        Demo
                    </button>
                    <button
                        className={targetMode === 'real' ? 'active' : ''}
                        onClick={() => handleModeChange('real')}
                        disabled={!isBotStopped}
                    >
                        Real
                    </button>
                </div>
            </div>

            <div className="status-container">
                <div className="status-item">
                    <span>Master Controller:</span>
                    <span className={`status-pill status-${masterStatus.master_mode}`}>{masterStatus.master_mode}</span>
                </div>
                {masterStatus.master_mode === 'running' && (
                    <>
                        <div className="status-item">
                            <span>Market State:</span>
                            <span className="status-value">{masterStatus.market_state || 'N/A'} (ADX: {masterStatus.adx_value?.toFixed(2) || 'N/A'})</span>
                        </div>
                        <div className="status-item">
                            <span>Active Signal Bot:</span>
                            <span className="status-value">{signalStatus.strategy_name || 'None'} ({signalStatus.mode})</span>
                        </div>
                    </>
                )}
            </div>
            <div className="button-group">
                <button onClick={handleStart} disabled={isLoading || !isBotStopped}>
                    ▶ Start Autonomous Mode
                </button>
                <button onClick={handleStop} disabled={isLoading || isBotStopped} className="stop-button">
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
