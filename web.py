from flask import Flask, request, jsonify

# Create a Flask application instance
app = Flask(__name__)

# Define a POST API endpoint
@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract data from the request (e.g., name and email)
        name = data.get('name')
        email = data.get('email')

        # Check if the required data exists
        if not name or not email:
            return jsonify({'error': 'Missing name or email'}), 400

        # Return a response as JSON
        return jsonify({
            'message': f"Received data: Name = {name}, Email = {email}"
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
