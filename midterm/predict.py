import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = './data/xgboost_model_threshold_0.28.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    threshold = model_data['threshold']
    logger.info(f"Model loaded successfully with threshold: {threshold}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def to_snake_case(name):
    """Convert column name to snake_case"""
    # Strip whitespace
    name = name.strip()
    # Replace special characters with underscores
    name = re.sub(r'[^\w\s-]', '_', name)
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s-]+', '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Convert to lowercase
    return name.lower()

FEATURE_NAMES = [
    'roa_c_before_interest_and_depreciation_before_interest',
    'roa_a_before_interest_and_after_tax',
    'roa_b_before_interest_and_depreciation_after_tax',
    'operating_gross_margin',
    'realized_sales_gross_margin',
    'operating_profit_rate',
    'pre_tax_net_interest_rate',
    'after_tax_net_interest_rate',
    'non_industry_income_and_expenditure_revenue',
    'continuous_interest_rate_after_tax',
    'operating_expense_rate',
    'research_and_development_expense_rate',
    'cash_flow_rate',
    'interest_bearing_debt_interest_rate',
    'tax_rate_a',
    'net_value_per_share_b',
    'net_value_per_share_a',
    'net_value_per_share_c',
    'persistent_eps_in_the_last_four_seasons',
    'cash_flow_per_share',
    'revenue_per_share_yuan',
    'operating_profit_per_share_yuan',
    'per_share_net_profit_before_tax_yuan',
    'realized_sales_gross_profit_growth_rate',
    'operating_profit_growth_rate',
    'after_tax_net_profit_growth_rate',
    'regular_net_profit_growth_rate',
    'continuous_net_profit_growth_rate',
    'total_asset_growth_rate',
    'net_value_growth_rate',
    'total_asset_return_growth_rate_ratio',
    'cash_reinvestment',
    'current_ratio',
    'quick_ratio',
    'interest_expense_ratio',
    'total_debt_total_net_worth',
    'debt_ratio',
    'net_worth_assets',
    'long_term_fund_suitability_ratio_a',
    'borrowing_dependency',
    'contingent_liabilities_net_worth',
    'operating_profit_paid_in_capital',
    'net_profit_before_tax_paid_in_capital',
    'inventory_and_accounts_receivable_net_value',
    'total_asset_turnover',
    'accounts_receivable_turnover',
    'average_collection_days',
    'inventory_turnover_rate_times',
    'fixed_assets_turnover_frequency',
    'net_worth_turnover_rate_times',
    'revenue_per_person',
    'operating_profit_per_person',
    'allocation_rate_per_person',
    'working_capital_to_total_assets',
    'quick_assets_total_assets',
    'current_assets_total_assets',
    'cash_total_assets',
    'quick_assets_current_liability',
    'cash_current_liability',
    'current_liability_to_assets',
    'operating_funds_to_liability',
    'inventory_working_capital',
    'inventory_current_liability',
    'current_liabilities_liability',
    'working_capital_equity',
    'current_liabilities_equity',
    'long_term_liability_to_current_assets',
    'retained_earnings_to_total_assets',
    'total_income_total_expense',
    'total_expense_assets',
    'current_asset_turnover_rate',
    'quick_asset_turnover_rate',
    'working_capitcal_turnover_rate',
    'cash_turnover_rate',
    'cash_flow_to_sales',
    'fixed_assets_to_assets',
    'current_liability_to_liability',
    'current_liability_to_equity',
    'equity_to_long_term_liability',
    'cash_flow_to_total_assets',
    'cash_flow_to_liability',
    'cfo_to_assets',
    'cash_flow_to_equity',
    'current_liability_to_current_assets',
    'liability_assets_flag',
    'net_income_to_total_assets',
    'total_assets_to_gnp_price',
    'no_credit_interval',
    'gross_profit_to_sales',
    'net_income_to_stockholder_s_equity',
    'liability_to_equity',
    'degree_of_financial_leverage_dfl',
    'interest_coverage_ratio_interest_expense_to_ebit',
    'net_income_flag',
    'equity_to_liability'
]
logger.info(f"Using {len(FEATURE_NAMES)} hardcoded feature names")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format. Expected a dictionary with feature names as keys'}), 400
        
        # Normalize incoming feature names to snake_case
        normalized_data = {to_snake_case(k): v for k, v in data.items()}
        
        missing_features = set(FEATURE_NAMES) - set(normalized_data.keys())
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}'
            }), 400
        
        input_data = pd.DataFrame([normalized_data])[FEATURE_NAMES]
        
        proba = model.predict_proba(input_data)[0, 1]
        prediction = 1 if proba >= threshold else 0
        
        logger.info(f"Prediction: {prediction}, Probability: {proba:.4f}")
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(proba),
            'threshold': float(threshold),
            'bankrupt': bool(prediction == 1)
        }), 200
            
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=False)

