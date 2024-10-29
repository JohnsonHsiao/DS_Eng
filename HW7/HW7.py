import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_and_process_data():
    data = pd.read_csv('amr_ds.csv')
    return data

def naive_bayes_classification(data):
    X = data[['Ampicillin', 'Penicillin']]
    y = data['Not_MDR']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

def calculate_transition_values(data):
    amp_pen = len(data[(data['Ampicillin'] == 1) & (data['Penicillin'] == 1)])
    amp_nmdr = len(data[(data['Ampicillin'] == 1) & (data['Not_MDR'] == 1)])
    pen_nmdr = len(data[(data['Penicillin'] == 1) & (data['Not_MDR'] == 1)])
    
    return amp_pen, amp_nmdr, pen_nmdr

def create_transition_matrix(amp_pen, amp_nmdr, pen_nmdr):
    transition_matrix = np.array([
        [0, amp_pen/(amp_nmdr+amp_pen), amp_nmdr/(amp_nmdr+amp_pen)],
        [amp_pen/(pen_nmdr+amp_pen), 0, pen_nmdr/(pen_nmdr+amp_pen)],
        [amp_nmdr/(amp_nmdr+pen_nmdr), pen_nmdr/(amp_nmdr+pen_nmdr), 0]
    ])
    return transition_matrix

def calculate_stationary_state(transition_matrix):
    eigenvals, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_index = np.argmin(np.abs(eigenvals - 1))
    stationary_state = eigenvectors[:, stationary_index].real
    stationary_state = stationary_state / np.sum(stationary_state)
    return stationary_state

def viterbi_algorithm(obs_sequence, states, transition_matrix, emission_probs):
    n_states = len(states)
    n_obs = len(obs_sequence)
    
    V = np.zeros((n_states, n_obs))
    path = np.zeros((n_states, n_obs), dtype=int)
    
    initial_prob = np.array([1/n_states] * n_states)
    
    for s in range(n_states):
        V[s, 0] = np.log(initial_prob[s]) + np.log(emission_probs[s][obs_sequence[0]])
    
    for t in range(1, n_obs):
        for s in range(n_states):
            prob_state = np.zeros(n_states)
            for s0 in range(n_states):
                prob_state[s0] = V[s0, t-1] + np.log(transition_matrix[s0, s]) + \
                                np.log(emission_probs[s][obs_sequence[t]])
            V[s, t] = np.max(prob_state)
            path[s, t] = np.argmax(prob_state)
    
    best_path = np.zeros(n_obs, dtype=int)
    best_path[-1] = np.argmax(V[:, -1])
    for t in range(n_obs-2, -1, -1):
        best_path[t] = path[int(best_path[t+1]), t+1]
    
    return [states[i] for i in best_path]

def main():
    data = load_and_process_data()
    
    accuracy = naive_bayes_classification(data)
    print(f"accuracy: {accuracy:.4f}")
    
    amp_pen, amp_nmdr, pen_nmdr = calculate_transition_values(data)
    print(f"\value:")
    print(f"amp_pen: {amp_pen}")
    print(f"amp_nmdr: {amp_nmdr}")
    print(f"pen_nmdr: {pen_nmdr}")
    
    transition_matrix = create_transition_matrix(amp_pen, amp_nmdr, pen_nmdr)
    print("\transition_matrix:")
    print(transition_matrix)
    
    stationary_state = calculate_stationary_state(transition_matrix)
    print("\stationary_state:")
    print(stationary_state)
    
    states = ['AMP', 'PEN', 'NMDR']
    obs_sequence = ['Infection', 'No Infection', 'Infection']
    
    emission_probs = {
        0: {'Infection': 0.6, 'No Infection': 0.4},  # AMP
        1: {'Infection': 0.7, 'No Infection': 0.3},  # PEN
        2: {'Infection': 0.2, 'No Infection': 0.8}   # NMDR
    }
    
    predicted_sequence = viterbi_algorithm(obs_sequence, states, transition_matrix, emission_probs)
    print("\predicted_sequence:")
    print(predicted_sequence)

if __name__ == "__main__":
    main()