import unittest
import json
import os
import sys

# Ensure backend path is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import app

class FlaskAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        """Test GET /health endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])

    def test_valid_prediction(self):
        """Test POST /predict with valid property inputs."""
        payload = {
            "Square_Footage": 2500,
            "Num_Bedrooms": 3,
            "Num_Bathrooms": 2,
            "Year_Built": 2005,
            "Lot_Size": 3.5,
            "Garage_Size": 2,
            "Neighborhood_Quality": 8
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('predicted_price', data)
        self.assertGreater(data['predicted_price'], 0)

    def test_missing_field(self):
        """Test POST /predict with missing required fields."""
        payload = {
            "Square_Footage": 2500,
            "Num_Bedrooms": 3
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Missing required feature fields', data['error'])

    def test_invalid_data_type(self):
        """Test POST /predict with non-numerical inputs."""
        payload = {
            "Square_Footage": "two thousand",
            "Num_Bedrooms": 3,
            "Num_Bathrooms": 2,
            "Year_Built": 2005,
            "Lot_Size": 3.5,
            "Garage_Size": 2,
            "Neighborhood_Quality": 8
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_out_of_range_neighborhood_quality(self):
        """Test POST /predict with Neighborhood_Quality out of 1-10 range."""
        payload = {
            "Square_Footage": 2500,
            "Num_Bedrooms": 3,
            "Num_Bathrooms": 2,
            "Year_Built": 2005,
            "Lot_Size": 3.5,
            "Garage_Size": 2,
            "Neighborhood_Quality": 15
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()
