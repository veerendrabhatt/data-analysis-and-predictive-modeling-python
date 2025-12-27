import requests
import sys
import json
from datetime import datetime

class DataScienceAPITester:
    def __init__(self, base_url="https://data-science-intern.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            
            result = {
                "test_name": name,
                "endpoint": endpoint,
                "method": method,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }

            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    result["response_data"] = response_data
                    # Print key info for successful responses
                    if "dataset_name" in response_data:
                        print(f"   Dataset: {response_data['dataset_name']}")
                    elif "summary" in response_data:
                        print(f"   Summary: {response_data['summary']}")
                    elif "models" in response_data:
                        print(f"   Models trained: {list(response_data['models'].keys())}")
                    elif "prediction" in response_data:
                        print(f"   Prediction: {response_data['prediction_formatted']}")
                except:
                    result["response_data"] = "Non-JSON response"
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    result["error_data"] = error_data
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    result["error_data"] = response.text[:200]
                    print(f"   Error: {response.text[:200]}")

            self.test_results.append(result)
            return success, response.json() if success else {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            result = {
                "test_name": name,
                "endpoint": endpoint,
                "method": method,
                "expected_status": expected_status,
                "actual_status": "ERROR",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(result)
            return False, {}

    def test_step_1_load_data(self):
        """Test Step 1: Load Dataset"""
        success, response = self.run_test(
            "Step 1: Load Dataset",
            "GET",
            "data/info",
            200
        )
        
        if success:
            # Validate response structure
            required_fields = ["dataset_name", "shape", "columns", "sample_data", "missing_values"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
            
            # Validate data structure
            if response["shape"]["rows"] <= 0 or response["shape"]["columns"] <= 0:
                print(f"⚠️  Warning: Invalid dataset shape: {response['shape']}")
                return False
                
            print(f"   ✓ Dataset loaded: {response['shape']['rows']} rows, {response['shape']['columns']} columns")
            return True
        return False

    def test_step_2_clean_data(self):
        """Test Step 2: Data Cleaning"""
        success, response = self.run_test(
            "Step 2: Clean Data",
            "GET",
            "data/clean",
            200
        )
        
        if success:
            # Validate cleaning results
            required_fields = ["original_shape", "cleaned_shape", "cleaning_steps"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
                
            print(f"   ✓ Data cleaned: {len(response['cleaning_steps'])} cleaning steps performed")
            return True
        return False

    def test_step_3_eda(self):
        """Test Step 3: Exploratory Data Analysis"""
        success, response = self.run_test(
            "Step 3: Perform EDA",
            "GET",
            "data/eda",
            200
        )
        
        if success:
            # Validate EDA results
            required_fields = ["summary_statistics", "correlation_matrix", "plots"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
                
            # Check if plots are base64 encoded
            plots = response.get("plots", {})
            if "correlation_heatmap" in plots and plots["correlation_heatmap"].startswith("data:image"):
                print(f"   ✓ Correlation heatmap generated successfully")
            if "distributions" in plots and plots["distributions"].startswith("data:image"):
                print(f"   ✓ Distribution plots generated successfully")
                
            return True
        return False

    def test_step_4_train_model(self):
        """Test Step 4: Model Training"""
        success, response = self.run_test(
            "Step 4: Train Model",
            "POST",
            "model/train",
            200
        )
        
        if success:
            # Validate model training results
            required_fields = ["models", "train_test_split", "plots"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
                
            models = response.get("models", {})
            if "linear_regression" in models and "random_forest" in models:
                lr_r2 = models["linear_regression"].get("r2", 0)
                rf_r2 = models["random_forest"].get("r2", 0)
                print(f"   ✓ Linear Regression R²: {lr_r2:.4f}")
                print(f"   ✓ Random Forest R²: {rf_r2:.4f}")
                
                if lr_r2 > 0.5 and rf_r2 > 0.5:
                    print(f"   ✓ Both models show good performance (R² > 0.5)")
                else:
                    print(f"   ⚠️  Warning: Low model performance detected")
                    
            return True
        return False

    def test_step_5_predict(self):
        """Test Step 5: Make Predictions"""
        # Test with sample California housing data
        sample_input = {
            "MedInc": 3.5,
            "HouseAge": 25.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.2,
            "Population": 1500.0,
            "AveOccup": 3.0,
            "Latitude": 37.5,
            "Longitude": -122.3
        }
        
        success, response = self.run_test(
            "Step 5: Make Prediction",
            "POST",
            "model/predict",
            200,
            data=sample_input
        )
        
        if success:
            # Validate prediction results
            required_fields = ["prediction", "prediction_formatted", "input_features"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                print(f"⚠️  Warning: Missing fields in response: {missing_fields}")
                return False
                
            prediction = response.get("prediction", 0)
            if prediction > 0:
                print(f"   ✓ Prediction generated: {response.get('prediction_formatted', 'N/A')}")
                return True
            else:
                print(f"   ⚠️  Warning: Invalid prediction value: {prediction}")
                
        return False

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        print(f"\n🔍 Testing Error Handling...")
        
        # Test accessing clean without loading data first
        try:
            response = requests.get(f"{self.api_url}/data/clean", timeout=10)
            if response.status_code == 400:
                print(f"   ✓ Proper error handling for premature data cleaning")
            else:
                print(f"   ⚠️  Expected 400 error for premature cleaning, got {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error testing error handling: {str(e)}")

        # Test prediction without training model
        try:
            sample_input = {"MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 5.5, "AveBedrms": 1.2, 
                          "Population": 1500.0, "AveOccup": 3.0, "Latitude": 37.5, "Longitude": -122.3}
            response = requests.post(f"{self.api_url}/model/predict", json=sample_input, timeout=10)
            if response.status_code == 400:
                print(f"   ✓ Proper error handling for prediction without training")
            else:
                print(f"   ⚠️  Expected 400 error for prediction without training, got {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error testing prediction error handling: {str(e)}")

def main():
    print("🚀 Starting Data Science API Testing...")
    print("=" * 60)
    
    tester = DataScienceAPITester()
    
    # Test the complete ML pipeline in sequence
    step1_success = tester.test_step_1_load_data()
    
    if step1_success:
        step2_success = tester.test_step_2_clean_data()
        
        if step2_success:
            step3_success = tester.test_step_3_eda()
            
            if step3_success:
                step4_success = tester.test_step_4_train_model()
                
                if step4_success:
                    step5_success = tester.test_step_5_predict()
                else:
                    print("❌ Skipping prediction test due to model training failure")
            else:
                print("❌ Skipping model training due to EDA failure")
        else:
            print("❌ Skipping EDA due to data cleaning failure")
    else:
        print("❌ Skipping all subsequent tests due to data loading failure")
    
    # Test error handling regardless of pipeline success
    tester.test_error_handling()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Test Results Summary:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    # Save detailed results
    results_file = "/app/test_reports/backend_api_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "tests_run": tester.tests_run,
                "tests_passed": tester.tests_passed,
                "success_rate": (tester.tests_passed/tester.tests_run)*100,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": tester.test_results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
