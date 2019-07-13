def MC_prediction_update_calc(cur_value, alpha, cur_return_value):
    '''
    Monte Carlo Prediction Update calculation
    
    V(s) := V(s) + alpha(G(s) - V(s))
    '''
    return cur_value + alpha * (cur_return_value - cur_value)


def TD_prediction_update_calc(cur_value, alpha, reward, discount_factor, next_value):
    '''
    Temporal Difference Prediction Update calculation
    
    V(s) := V(s) + alpha(R + γV(s_t+1)- V(s))
    '''
    return cur_value + alpha * (reward +  discount_factor*next_value - cur_value)


def start_training():
    '''
    episode 정할 때 learning rate를 1/episode로
    '''
