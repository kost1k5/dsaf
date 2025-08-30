import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <div className="logo">
        <h2>Kost1kTrade</h2>
        <p>Nebula Command</p>
      </div>
      <nav>
        <ul>
          <li><NavLink to="/" end>Dashboard</NavLink></li>
          <li><NavLink to="/strategies">Strategies</NavLink></li>
          {/* These are placeholders for now */}
          <li><a href="#" className="disabled-link">Backtesting</a></li>
          <li><a href="#" className="disabled-link">Settings</a></li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
