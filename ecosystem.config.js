module.exports = {
  apps: [
    {
      name: 'kost1ktrade-backend',
      script: 'pipenv',
      args: 'run uvicorn src.api.main:app --host 0.0.0.0 --port 8000',
      cwd: './kost1ktrade/backend',
      watch: false,
      interpreter: 'none',
      windowsHide: true,
    },
    {
      name: 'kost1ktrade-frontend',
      script: './node_modules/vite/bin/vite.js',
      args: 'dev',
      cwd: './kost1ktrade/frontend',
      watch: false,
      interpreter: 'node',
      windowsHide: true,
    },
  ],
};
