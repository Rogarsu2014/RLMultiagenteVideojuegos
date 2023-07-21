def calcularQ(actualQ, alpha, reward, gamma, max_next_state):
    return (1 - alpha) * actualQ + alpha * (reward + gamma * max_next_state)
