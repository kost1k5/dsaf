import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import StrategyManager from './pages/StrategyManager';
import SimulationDeck from './pages/SimulationDeck';
import './App.css';

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/strategies" element={<StrategyManager />} />
            <Route path="/simulations" element={<SimulationDeck />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
