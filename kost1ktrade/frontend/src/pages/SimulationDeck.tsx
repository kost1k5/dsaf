import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './SimulationDeck.css';

// --- TYPE DEFINITIONS ---
type StrategyParams = { [key: string]: number | string };

type StrategyConfig = {
  selected: boolean;
  params: StrategyParams;
};

type StrategiesState = {
  [strategyName: string]: StrategyConfig;
};

type SimulationResult = {
    strategy_name: string;
    params: StrategyParams;
    metrics: { [key: string]: number | string };
    error?: string;
};

// --- HELPER FUNCTIONS ---
const toTitleCase = (str: string) => {
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};


// --- SUB-COMPONENTS ---
const ResultsTable = ({ results }: { results: SimulationResult[] }) => {
    if (!results || results.length === 0) {
        return <p>No results to display.</p>;
    }

    // Find the first result that has metrics to determine the headers
    const firstResultWithMetrics = results.find(r => r.metrics);
    if (!firstResultWithMetrics) {
        return <p>Simulation ran, but no valid results were returned. Check the error messages below.</p>;
    }
    const headers = Object.keys(firstResultWithMetrics.metrics);

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
                    <tr key={index} className={result.error ? 'error-row' : ''}>
                        <td>{toTitleCase(result.strategy_name)}</td>
                        {result.error ? (
                            <td colSpan={headers.length} className="error-message">{result.error}</td>
                        ) : (
                            headers.map(header => (
                                <td key={header}>
                                    {typeof result.metrics[header] === 'number'
                                        ? (result.metrics[header] as number).toFixed(4)
                                        : result.metrics[header]}
                                </td>
                            ))
                        )}
                    </tr>
                ))}
            </tbody>
        </table>
    );
};


// --- MAIN COMPONENT ---
const SimulationDeck = () => {
    // State for form inputs
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [timeframe, setTimeframe] = useState('1h');
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2023-03-31');

    // State for strategies
    const [strategies, setStrategies] = useState<StrategiesState>({});

    // State for UI feedback
    const [isLoading, setIsLoading] = useState(false);
    const [feedback, setFeedback] = useState({ type: '', message: '' });
    const [results, setResults] = useState<SimulationResult[] | null>(null);

    // Fetch available strategies on component mount
    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response = await axios.get('/api/strategies/status');
                const availableStrategies: { [key: string]: { params: StrategyParams } } = response.data;
                const initialStrategiesState: StrategiesState = {};
                for (const name in availableStrategies) {
                    initialStrategiesState[name] = {
                        selected: false, // Start with all strategies deselected
                        params: availableStrategies[name].params,
                    };
                }
                setStrategies(initialStrategiesState);
            } catch (error) {
                console.error("Failed to fetch strategies", error);
                setFeedback({ type: 'error', message: 'Could not load strategies from server.' });
            }
        };
        fetchStrategies();
    }, []);

    // --- EVENT HANDLERS ---

    const handleStrategyToggle = (strategyName: string) => {
        setStrategies(prev => ({
            ...prev,
            [strategyName]: {
                ...prev[strategyName],
                selected: !prev[strategyName].selected,
            },
        }));
    };

    const handleParamChange = (strategyName: string, paramName: string, value: string) => {
        const isNumeric = !isNaN(Number(value)) && value !== '';
        setStrategies(prev => ({
            ...prev,
            [strategyName]: {
                ...prev[strategyName],
                params: {
                    ...prev[strategyName].params,
                    [paramName]: isNumeric ? parseFloat(value) : value,
                },
            },
        }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setFeedback({ type: '', message: '' });
        setResults(null);

        const selectedStrategies = Object.entries(strategies)
            .filter(([, config]) => config.selected)
            .map(([name, config]) => ({ name, params: config.params }));

        if (selectedStrategies.length === 0) {
            setFeedback({ type: 'error', message: 'Please select at least one strategy to run.' });
            setIsLoading(false);
            return;
        }

        const requestBody = {
            symbol,
            timeframe,
            start_date: startDate,
            end_date: endDate,
            strategies: selectedStrategies,
        };

        try {
            const response = await axios.post('/api/simulation/run', requestBody);
            setResults(response.data);
            setFeedback({ type: 'success', message: 'Simulation completed!' });
        } catch (err: any) {
            setFeedback({ type: 'error', message: err.response?.data?.detail || 'Failed to run simulation.' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="simulation-deck">
            <h2>Simulation Deck</h2>
            <p>Select and configure strategies to run a comparative backtest.</p>

            <form onSubmit={handleSubmit} className="simulation-form">
                {/* --- Column 1: General Settings --- */}
                <div className="form-column">
                    <fieldset className="strategy-config">
                        <legend>General Settings</legend>
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
                    </fieldset>
                </div>

                {/* --- Column 2: Strategy Selection & Configuration --- */}
                <div className="form-column">
                     <fieldset className="strategy-config">
                        <legend>Select Strategies</legend>
                        <div className="strategy-list">
                            {Object.entries(strategies).map(([name, config]) => (
                                <div key={name} className="strategy-item">
                                    <div className="strategy-toggle">
                                        <input
                                            type="checkbox"
                                            id={`strategy-${name}`}
                                            checked={config.selected}
                                            onChange={() => handleStrategyToggle(name)}
                                        />
                                        <label htmlFor={`strategy-${name}`}>{toTitleCase(name)}</label>
                                    </div>
                                    {config.selected && (
                                        <div className="strategy-params">
                                            {Object.entries(config.params).map(([paramName, paramValue]) => (
                                                <div className="param-group" key={paramName}>
                                                    <label htmlFor={`${name}-${paramName}`}>{toTitleCase(paramName)}</label>
                                                    <input
                                                        type="number"
                                                        id={`${name}-${paramName}`}
                                                        value={paramValue}
                                                        onChange={(e) => handleParamChange(name, paramName, e.target.value)}
                                                        step="any"
                                                    />
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
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

            {results && results.length > 0 && (
                <div className="results-container">
                    <h3>Simulation Results</h3>
                    <ResultsTable results={results} />
                </div>
            )}
        </div>
    );
};

export default SimulationDeck;
