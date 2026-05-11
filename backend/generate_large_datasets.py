"""Generate larger training datasets with 500 and 1000 samples"""
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_circuit_data(n_samples, filename):
    """Generate synthetic circuit fault data and save to CSV"""
    
    # Normal circuits (fault=0)
    n_normal = n_samples // 2
    normal_voltage = np.random.normal(230, 5, n_normal)
    normal_current = np.random.normal(10, 1, n_normal)
    normal_temp = np.random.normal(35, 3, n_normal)
    normal_resistance = np.random.normal(23, 2, n_normal)
    normal_frequency = np.random.normal(50, 0.5, n_normal)
    normal_power = normal_voltage * normal_current
    
    normal_data = np.column_stack([
        normal_voltage, normal_current, normal_temp, 
        normal_resistance, normal_frequency, normal_power
    ])
    normal_labels = np.zeros(n_normal)
    
    # Faulty circuits (fault=1)
    n_faulty = n_samples - n_normal
    faulty_voltage = np.random.normal(240, 15, n_faulty)
    faulty_current = np.random.normal(15, 3, n_faulty)
    faulty_temp = np.random.normal(55, 8, n_faulty)
    faulty_resistance = np.random.normal(20, 3, n_faulty)
    faulty_frequency = np.random.normal(49, 1.5, n_faulty)
    faulty_power = faulty_voltage * faulty_current
    
    faulty_data = np.column_stack([
        faulty_voltage, faulty_current, faulty_temp,
        faulty_resistance, faulty_frequency, faulty_power
    ])
    faulty_labels = np.ones(n_faulty)
    
    # Combine
    X = np.vstack([normal_data, faulty_data])
    y = np.hstack([normal_labels, faulty_labels])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['voltage', 'current', 'temperature', 'resistance', 'frequency', 'power'])
    df['fault'] = y
    
    # Round to 2 decimal places for readability
    df = df.round(2)
    
    # Save
    df.to_csv(filename, index=False)
    print(f"✓ Created {filename} with {len(df)} samples")

# Generate files
base_path = r"c:\Users\sriha\Downloads\FINDMYNEST\New folder\circuit-fault-detection\backend\sample_data"

generate_circuit_data(500, f"{base_path}\\training_data_500.csv")
generate_circuit_data(1000, f"{base_path}\\training_data_1000.csv")

print("\nDatasets created successfully!")
print("You can now upload these files to your website to test accuracy with different dataset sizes.")
