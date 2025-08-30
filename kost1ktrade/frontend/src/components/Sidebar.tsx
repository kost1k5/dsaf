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
          <li className="active"><a href="#">Dashboard</a></li>
          <li><a href="#">Strategies</a></li>
          <li><a href="#">Backtesting</a></li>
          <li><a href="#">Settings</a></li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
