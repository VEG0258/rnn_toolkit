import matplotlib.pyplot as plt
import numpy as np

def stability_checker(hidden_states_h, hidden_states_c, time_steps):
    plt.figure(figsize=(14, 8))
    
    # Plot h_t (hidden states)
    plt.subplot(2, 1, 1)
    plt.plot(hidden_states_h[:, :time_steps])
    plt.title('Hidden States h_t over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Hidden State Value')
    
    # Plot c_t (cell states)
    plt.subplot(2, 1, 2)
    plt.plot(hidden_states_c[:, :time_steps])
    plt.title('Cell States c_t over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cell State Value')
    
    plt.tight_layout()
    plt.savefig(f"../tests/stability_plot.png")
    plt.close()
    print("stability plot printed")