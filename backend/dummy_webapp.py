# dummy_webapp.py
from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

# Dummy credentials (in-memory, for demo purposes only)
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# HTML template for login page
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        .login-box { width: 300px; margin: 0 auto; padding: 20px; border: 1px solid #ccc; }
        input { width: 100%; margin: 10px 0; padding: 5px; }
        button { padding: 5px 10px; }
    </style>
</head>
<body>
    <div class="login-box">
        <h2>Login</h2>
        <form method="post" action="/login">
            <input type="text" name="username" placeholder="Username" required><br>
            <input type="password" name="password" placeholder="Password" required><br>
            <button type="submit">Login</button>
        </form>
        {% if error %}
            <p style="color: red;">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    # Redirect to login if not authenticated (simple session check)
    if 'username' not in request.args:
        return redirect('/login')
    return "Welcome to Dummy Web App! You are logged in as admin."

@app.route('/simulate_attack')
def simulate_attack():
    # Redirect to login if not authenticated
    if 'username' not in request.args:
        return redirect('/login')
    return "Simulated Attack Endpoint"

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            # Redirect to homepage with a query param to simulate a logged-in state
            return redirect('/?username=admin')
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template_string(LOGIN_TEMPLATE, error=error)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)