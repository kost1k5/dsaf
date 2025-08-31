import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './SimulationDeck.css';

// Types to match the backend
type StrategyParams = { [key: string]: number | string };
type AvailableStrategies = { [key: string]: StrategyParams };
type SimulationResult = {
    strategy_name: string;
    params: StrategyParams;
    metrics: { [key: string]: number | string };
    error?: string;
};

// Helper function to format snake_case to Title Case
const toTitleCase = (str: string) => {
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

// Results Table Component
const ResultsTable = ({ results }: { results: SimulationResult[] }) => {
    if (!results || results.length === 0) {
        return <p>No results to display.</p>;
    }

    // Assuming all results have the same metrics keys
    const headers = Object.keys(results[0].metrics);

    return (
        <table className="results-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    {headers.map(header => <th key={header}>{toTitleCase(header)}</th>)}
                </tr>
            </thead>
            <tbody>
                {results.map((result, index) => (
                    <tr key={index}>
                        <td>{toTitleCase(result.strategy_name)}</td>
                        {headers.map(header => (
                            <td key={header}>
                                {typeof result.metrics[header] === 'number'
                                    ? (result.metrics[header] as number).toFixed(4)
                                    : result.metrics[header]}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

const SimulationDeck = () => {
    // State for API data
    const [availableStrategies, setAvailableStrategies] = useState<AvailableStrategies>({});

    // State for form inputs
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [timeframe, setTimeframe] = useState('1h');
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2023-03-31');
    const [selectedStrategy, setSelectedStrategy] = useState('');
    const [strategyParams, setStrategyParams] = useState<StrategyParams>({});

    // State for UI feedback
    const [isLoading, setIsLoading] = useState(false);
    const [feedback, setFeedback] = useState({ type: '', message: '' });
    const [results, setResults] = useState<SimulationResult[] | null>(null);

    // Fetch available strategies on component mount
    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response = await axios.get('/api/strategies');
                setAvailableStrategies(response.data);
                const firstStrategyName = Object.keys(response.data)[0];
                if (firstStrategyName) {
                    setSelectedStrategy(firstStrategyName);
                    setStrategyParams(response.data[firstStrategyName]);
                }
            } catch (error) {
                console.error("Failed to fetch strategies", error);
                setFeedback({ type: 'error', message: 'Could not load strategies from server.' });
            }
        };
        fetchStrategies();
    }, []);

    // Handler for strategy selection change
    const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newStrategyName = e.target.value;
        setSelectedStrategy(newStrategyName);
        setStrategyParams(availableStrategies[newStrategyName] || {});
    };

    // Handler for strategy parameter changes
    const handleParamChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        const isNumeric = !isNaN(Number(value)) && value !== '';
        setStrategyParams({
            ...strategyParams,
            [name]: isNumeric ? parseFloat(value) : value,
        });
    };

    // Form submission handler
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setFeedback({ type: '', message: '' });
        setResults(null);

        const requestBody = {
            symbol,
            timeframe,
            start_date: startDate,
            end_date: endDate,
            strategies: [
                {
                    name: selectedStrategy,
                    params: strategyParams,
                },
            ],
        };

        try {
            const response = await axios.post('/api/simulation/run', requestBody);
            // Check for errors within the results
            const hasError = response.data.some((res: SimulationResult) => res.error);
            if (hasError) {
                const errorMsg = response.data.find((res: SimulationResult) => res.error).error;
                setFeedback({ type: 'error', message: `Simulation failed: ${errorMsg}` });
            } else {
                setResults(response.data);
                setFeedback({ type: 'success', message: 'Simulation completed successfully!' });
            }
        } catch (err: any) {
            setFeedback({ type: 'error', message: err.response?.data?.detail || 'Failed to run simulation.' });
        } finally {
            setIsLoading(false);
        }
    };

    // Render dynamic inputs for strategy parameters
    const renderStrategyParams = () => {
        if (!selectedStrategy) return null;
        return Object.entries(strategyParams).map(([paramName, paramValue]) => (
            <div className="form-group" key={paramName}>
                <label htmlFor={paramName}>{toTitleCase(paramName)}</label>
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
        <div className="simulation-deck">
            <h2>Simulation Deck</h2>
            <p>Configure and run a new backtest simulation.</p>

            <form onSubmit={handleSubmit} className="simulation-form">
                <div className="form-column">
                    <div className="form-group">
                        <label htmlFor="symbol">Symbol</label>
                        <input type="text" id="symbol" value={symbol} onChange={e => setSymbol(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label htmlFor="timeframe">Timeframe</label>
                        <input type="text" id="timeframe" value={timeframe} onChange={e => setTimeframe(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label htmlFor="start-date">Start Date</label>
                        <input type="date" id="start-date" value={startDate} onChange={e => setStartDate(e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label htmlFor="end-date">End Date</label>
                        <input type="date" id="end-date" value={endDate} onChange={e => setEndDate(e.target.value)} />
                    </div>
                </div>

                <div className="form-column">
                    <div className="form-group">
                        <label htmlFor="strategy-select">Strategy</label>
                        <select id="strategy-select" value={selectedStrategy} onChange={handleStrategyChange}>
                            {Object.keys(availableStrategies).map(name => (
                                <option key={name} value={name}>{toTitleCase(name)}</option>
                            ))}
                        </select>
                    </div>
                    <fieldset className="strategy-config">
                        <legend>Strategy Parameters</legend>
                        {renderStrategyParams()}
                    </fieldset>
                </div>

                <button type="submit" className="run-button" disabled={isLoading}>
                    {isLoading ? 'Running Simulation...' : 'Run Simulation'}
                </button>
            </form>

            {feedback.message && (
                <div className={`feedback-message ${feedback.type}`}>
                    {feedback.message}
                </div>
            )}

            {results && (
                <div className="results-container">
                    <h3>Simulation Results</h3>
                    <ResultsTable results={results} />
                </div>
            )}
        </div>
    );
};

export default SimulationDeck;
