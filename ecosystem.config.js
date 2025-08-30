module.exports = {
  apps: [
    {
      name: 'kost1ktrade-backend',
      script: 'pipenv',
      args: 'run uvicorn src.api.main:app --host 0.0.0.0 --port 8000',
      cwd: './kost1ktrade/backend',
      watch: false,
      interpreter: 'none', // Important for running non-node scripts
    },
    {
      name: 'kost1ktrade-frontend',
      script: 'npm',
      args: 'run dev',
      cwd: './kost1ktrade/frontend',
      watch: false,
      interpreter: 'none',
    },
  ],
};
