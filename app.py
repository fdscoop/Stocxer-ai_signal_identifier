from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ta
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def welcome():
    """Welcome page route"""
    return jsonify({
        "message": "Welcome to Stocxer AI - Your Advanced Options Analytics Platform",
        "version": "1.0.0",
        "status": "active",
        "strategy": "Stocxer-AI Signal Identifier"
    })

class NiftySignalGenerator:
    def __init__(self, lookback_period=14):
        self.lookback_period = lookback_period
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """
        Create technical indicators and candlestick patterns as features
        """
        df_features = df.copy()
        
        print(f"Initial data shape: {df_features.shape}")
        
        # Add trend indicators
        df_features['sma'] = ta.trend.sma_indicator(df_features['close'], 
                                                  window=min(self.lookback_period, len(df_features)-1))
        df_features['ema'] = ta.trend.ema_indicator(df_features['close'], 
                                                  window=min(self.lookback_period, len(df_features)-1))
        df_features['macd'] = ta.trend.macd_diff(df_features['close'])
        
        # Add momentum indicators
        df_features['rsi'] = ta.momentum.rsi(df_features['close'], 
                                           window=min(self.lookback_period, len(df_features)-1))
        df_features['stoch'] = ta.momentum.stoch(df_features['high'], 
                                               df_features['low'], 
                                               df_features['close'],
                                               window=min(self.lookback_period, len(df_features)-1))
        
        # Add volatility indicators
        df_features['bb_high'] = ta.volatility.bollinger_hband(df_features['close'],
                                                             window=min(self.lookback_period, len(df_features)-1))
        df_features['bb_low'] = ta.volatility.bollinger_lband(df_features['close'],
                                                           window=min(self.lookback_period, len(df_features)-1))
        df_features['atr'] = ta.volatility.average_true_range(df_features['high'], 
                                                            df_features['low'], 
                                                            df_features['close'],
                                                            window=min(self.lookback_period, len(df_features)-1))
        
        # Candlestick patterns
        df_features['body'] = df_features['close'] - df_features['open']
        df_features['upper_shadow'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
        df_features['lower_shadow'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
        
        # Add additional features
        df_features['daily_return'] = df_features['close'].pct_change()
        df_features['volatility'] = df_features['daily_return'].rolling(window=min(self.lookback_period, len(df_features)-1)).std()
        
        # Price levels relative to recent history
        lookback = min(self.lookback_period, len(df_features)-1)
        df_features['dist_from_high'] = df_features['close'] / df_features['high'].rolling(window=lookback).max() - 1
        df_features['dist_from_low'] = df_features['close'] / df_features['low'].rolling(window=lookback).min() - 1
        
        # Forward fill NaN values for a small number of steps
        df_features = df_features.fillna(method='ffill', limit=3)
        
        # Remove remaining NaN values
        df_features = df_features.dropna()
        
        print(f"Final data shape after feature preparation: {df_features.shape}")
        return df_features

    def fit_predict(self, df):
        """Train on initial data and generate predictions"""
        if len(df) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} data points, but got {len(df)}")
            
        print(f"Input data shape: {df.shape}")
        features_df = self.prepare_features(df)
        
        if len(features_df) == 0:
            raise ValueError("No valid data points after feature preparation")
            
        # Create labels based on future returns
        features_df['next_return'] = features_df['close'].pct_change().shift(-1)
        labels = np.zeros(len(features_df))
        labels[features_df['next_return'] > 0.001] = 1  # Buy signal
        labels[features_df['next_return'] < -0.001] = -1  # Sell signal
        
        # Remove NaN values
        valid_indices = ~np.isnan(labels)
        features_df = features_df[valid_indices]
        labels = labels[valid_indices]
        
        if len(features_df) == 0:
            raise ValueError("No valid data points after label creation")
            
        print(f"Data shape after label creation: {features_df.shape}")
        
        # Prepare features
        feature_columns = [col for col in features_df.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume', 'next_return']]
        X = features_df[feature_columns]
        
        print(f"Feature columns: {feature_columns}")
        print(f"Features shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, labels)
        
        # Generate predictions for all data
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities, features_df.index

def process_candlestick_data(data):
    """Convert input data to DataFrame"""
    df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

@app.route('/analyze-candlestick-data', methods=['POST'])
def analyze_candlestick_data():
    try:
        request_data = request.json
        if not request_data or 'data' not in request_data:
            return jsonify({
                'status': 'error',
                'message': 'Missing data field in request JSON'
            }), 400
            
        data = request_data['data']
        
        # Convert data to DataFrame
        df = process_candlestick_data(data)
        
        # Initialize model and generate predictions
        signal_generator = NiftySignalGenerator()
        predictions, probabilities, dates = signal_generator.fit_predict(df)
        
        # Prepare response
        results = []
        for date, pred, probs in zip(dates, predictions, probabilities):
            current_price = float(df.loc[date, 'close'])
            results.append({
                'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
                'price': current_price,
                'signal': int(pred),
                'probabilities': {
                    'sell': float(probs[0]),
                    'hold': float(probs[1]) if len(probs) > 2 else 0,
                    'buy': float(probs[-1])
                }
            })
        
        return jsonify({
            'status': 'success',
            'signals': results,
            'summary': {
                'total_signals': len(predictions),
                'buy_signals': int(sum(predictions == 1)),
                'sell_signals': int(sum(predictions == -1)),
                'hold_signals': int(sum(predictions == 0))
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)