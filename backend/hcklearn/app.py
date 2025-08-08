from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "change_this_to_a_secure_random_key"

# Dummy credentials
VALID_STUDENT_ID = "20250001"
VALID_PASSWORD   = "password123"

@app.route('/')
def index():
    if not session.get('student_id'):
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        sid = request.form['student_id']
        pwd = request.form['password']
        if sid == VALID_STUDENT_ID and pwd == VALID_PASSWORD:
            session['student_id'] = sid
            session['student_name'] = "Emmanuel"  # in real life, look up in DB
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid Student ID or Password."
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if not session.get('student_id'):
        return redirect(url_for('login'))
    return render_template(
        'dashboard.html',
        student_name=session.get('student_name')
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# (You can add the actual Registration & Check‑Result views here)

if __name__ == '__main__':
    app.run(debug=True)
