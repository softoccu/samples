import random
import pandas as pd

# Step 1: Define error types
error_types = [
    "NetworkError", 
    "DatabaseError", 
    "PermissionError", 
    "TimeoutError", 
    "LogicError", 
    "TokenExpired", 
    "CredentialExpired", 
    "PasswordNotMatch", 
    "ThisUserNotExist"
]

# Step 2: Define error log templates for each error type
error_templates = {
    "NetworkError": [
        "Network connection failed",
        "Unable to reach the server",
        "Network timeout occurred",
        "Cannot establish a network connection"
    ],
    "DatabaseError": [
        "Failed to connect to database",
        "Database query failed",
        "SQL syntax error",
        "Database connection lost"
    ],
    "PermissionError": [
        "Access denied",
        "Permission error",
        "Unauthorized access attempt",
        "You do not have permission to perform this action"
    ],
    "TimeoutError": [
        "Request timed out",
        "Operation took too long",
        "Server response delayed",
        "Timeout occurred while fetching data"
    ],
    "LogicError": [
        "Unexpected error occurred",
        "Null pointer exception",
        "Variable out of range",
        "Infinite loop detected"
    ],
    "TokenExpired": [
        "Token has expired, please login again",
        "Authentication token expired",
        "Session token invalid",
        "Token expired. Re-authentication required"
    ],
    "CredentialExpired": [
        "User credentials expired",
        "Expired credentials, please update",
        "Your login credentials have expired",
        "Credential expiration detected"
    ],
    "PasswordNotMatch": [
        "Password does not match",
        "Incorrect password entered",
        "Failed login attempt due to wrong password",
        "Password mismatch error"
    ],
    "ThisUserNotExist": [
        "User does not exist",
        "No user found with the given username",
        "Invalid username, user does not exist",
        "User is not registered in the system"
    ]
}

def generate_log_entry():
    log = []
    labels = {}

    # 随机选择 1~3 个标签
    selected_errors = random.sample(list(error_templates.items()), random.randint(1, 3))
    weights = [random.uniform(0.5, 2.0) for _ in selected_errors]
    total = sum(weights)
    probs = [round(w / total, 3) for w in weights]  # 概率加起来是 1.0

    for (error_type, templates), prob in zip(selected_errors, probs):
        log.append(random.choice(templates))
        labels[error_type] = prob

    log_text = " | ".join(log)
    return log_text, labels

# Generate a sample of 100 entries
data = []
for _ in range(100):
    log_text, labels = generate_log_entry()
    row = {"log": log_text}
    row.update(labels)  # Add the labels to the row
    data.append(row)

# Step 4: Create DataFrame
df = pd.DataFrame(data)

# Save to CSV if needed
df.to_csv("simulated_error_logs.csv", index=False)

# Display first few rows to verify
print(df.head())
