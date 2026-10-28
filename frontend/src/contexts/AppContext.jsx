import React, { createContext, useContext, useState } from 'react';

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState(null);
  const [taskType, setTaskType] = useState(null);
  const [preprocessingConfig, setPreprocessingConfig] = useState({
    numeric_imputer: 'median',
    numeric_scaler: 'standard',
    categorical_imputer: 'most_frequent',
    categorical_encoding: 'onehot',
    test_size: 0.2,
    random_seed: 42,
    drop_identifiers: true,
    exclude_columns: [],
    identifier_columns: [],
    handle_constant: true,
    time_series_date_column: '',
    time_series_lags: [1, 7, 14],
    time_series_rolling_windows: [3, 7],
  });
  const [selectedExperiments, setSelectedExperiments] = useState([]);

  const value = {
    selectedDataset, setSelectedDataset,
    targetColumn, setTargetColumn,
    taskType, setTaskType,
    preprocessingConfig, setPreprocessingConfig,
    selectedExperiments, setSelectedExperiments,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be inside AppProvider');
  return ctx;
};
