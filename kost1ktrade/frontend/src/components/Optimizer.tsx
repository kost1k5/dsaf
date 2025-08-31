import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Optimizer = () => {
    const [strategies, setStrategies] = useState([]);
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const [paramGrid, setParamGrid] = useState('');
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [timeframe, setTimeframe] = useState('1h');
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2023-03-31');
    const [optimizeFor, setOptimizeFor] = useState('sharpe_ratio');
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response = await axios.get('/api/strategies/status');
                // Filter for signal strategies, as we can't optimize grid bot yet
                const signalStrategies = Object.entries(response.data)
                    .filter(([, info]) => info.type === 'signal')
                    .map(([name]) => name);
                setStrategies(signalStrategies);
                if (signalStrategies.length > 0) {
                    setSelectedStrategy(signalStrategies[0]);
                }
            } catch (err) {
                console.error("Failed to fetch strategies", err);
                setError('Could not load strategies.');
            }
        };
        fetchStrategies();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        setResults(null);

        let parsedGrid;
        try {
            parsedGrid = JSON.parse(paramGrid);
        } catch (err) {
            setError('Invalid JSON in parameter grid.');
            setIsLoading(false);
            return;
        }

        const requestBody = {
            strategy_name: selectedStrategy,
            symbol,
            timeframe,
            start_date: startDate,
            end_date: endDate,
            param_grid: parsedGrid,
            optimize_for: optimizeFor,
        };

        try {
            const response = await axios.post('/api/optimization/run', requestBody);
            setResults(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to run optimization.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="optimizer-widget">
            <h3>Strategy Optimizer</h3>
            <form onSubmit={handleSubmit}>
                {/* Strategy Selector */}
                <select value={selectedStrategy} onChange={(e) => setSelectedStrategy(e.target.value)}>
                    {strategies.map(s => <option key={s} value={s}>{s}</option>)}
                </select>

                {/* Parameter Grid */}
                <textarea
                    value={paramGrid}
                    onChange={(e) => setParamGrid(e.target.value)}
                    placeholder='e.g., {"short_window": [10, 20], "long_window": [50, 100]}'
                />

                {/* Other inputs */}
                <input type="text" value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol" />
                <input type="text" value={timeframe} onChange={(e) => setTimeframe(e.target.value)} placeholder="Timeframe" />
                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                <input type="text" value={optimizeFor} onChange={(e) => setOptimizeFor(e.target.value)} placeholder="Optimize For" />

                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Optimizing...' : 'Run Optimization'}
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}

            {results && (
                <div className="optimizer-results">
                    <h4>Best Parameters:</h4>
                    <pre>{JSON.stringify(results.best_params, null, 2)}</pre>
                    <h4>Metrics:</h4>
                    <pre>{JSON.stringify(results.best_metrics, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default Optimizer;
