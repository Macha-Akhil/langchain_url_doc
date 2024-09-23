from flask import Flask, request, jsonify
import jwt as jwt_app
import datetime
from functools import wraps

app = Flask(__name__)

# Secret key (use environment variables in real applications)
app.config['SECRET_KEY'] = 'your_secret_key'


# Helper function to require a valid token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            # Decode and verify the token
            token = token.split(" ")[1]  # Extract Bearer token part
            data = jwt_app.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except jwt_app.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt_app.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(*args, **kwargs)

    return decorated


# Route to generate JWT token (login route)
@app.route('/login', methods=['POST'])
def login():
    auth_data = request.get_json()

    username = auth_data.get('username')
    password = auth_data.get('password')

    # In real applications, you'd verify the username and password with a database
    if username == 'user' and password == 'password':  # Dummy credentials check
        # Generate token valid for 1 hour
        token = jwt_app.encode({
            'username': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm="HS256")

        return jsonify({'token': token})

    return jsonify({'message': 'Invalid credentials'}), 401


# Protected route that requires JWT token
@app.route('/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({'message': 'This is a protected route. You have a valid token!'})


# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)

