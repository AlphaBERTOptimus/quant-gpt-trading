def _meets_criteria(self, stock_data: Dict, criteria: Dict) -> bool:
    """Check if stock meets screening criteria"""
    # PE ratio criteria
    if 'pe_max' in criteria and stock_data.get('pe_ratio'):
        if stock_data['pe_ratio'] > criteria['pe_max']:
            return False
    
    if 'pe_min' in criteria and stock_data.get('pe_ratio'):
        if stock_data['pe_ratio'] < criteria['pe_min']:
            return False
    
    # Market cap criteria
    if 'market_cap_min' in criteria and stock_data.get('market_cap'):
        if stock_data['market_cap'] < criteria['market_cap_min']:
            return False
    
    # RSI criteria
    if 'rsi_max' in criteria and stock_data.get('rsi'):
        if stock_data['rsi'] > criteria['rsi_max']:
            return False
    
    if 'rsi_min' in criteria and stock_data.get('rsi'):
        if stock_data['rsi'] < criteria['rsi_min']:
            return False

    # Dividend yield criteria
    if 'dividend_yield_min' in criteria and stock_data.get('dividend_yield'):
        if stock_data['dividend_yield'] < criteria['dividend_yield_min']:
            return False
    
    # High volume criteria
    if criteria.get('high_volume') and stock_data.get('volume'):
        # Consider high volume as > 1M shares
        if stock_data['volume'] < 1000000:
            return False
    
    return True
