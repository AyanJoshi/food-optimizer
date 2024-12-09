import requests
import json
import time
import os
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

class RecipeDatabase:
    DATABASE_FILE = "recipe_database.json"
    
    def __init__(self):
        self.recipes = {}  # Dictionary with recipe_id as key
        self.metadata = {
            'last_updated': '',
            'total_recipes': 0,
            'cuisine_distribution': {},
            'diet_distribution': {},
            'dish_type_distribution': {}
        }
        self.load_database()
    
    def load_database(self):
        """Load existing database if it exists"""
        if os.path.exists(self.DATABASE_FILE):
            try:
                with open(self.DATABASE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recipes = {str(recipe['id']): recipe for recipe in data.get('recipes', [])}
                    self.metadata = data.get('metadata', self.metadata)
                print(f"Loaded {len(self.recipes)} existing recipes from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.backup_corrupted_database()
    
    def backup_corrupted_database(self):
        """Create backup of corrupted database file"""
        if os.path.exists(self.DATABASE_FILE):
            backup_name = f"corrupted_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(self.DATABASE_FILE, backup_name)
            print(f"Created backup of corrupted database: {backup_name}")
    
    def add_recipe(self, recipe: Dict) -> bool:
        """Add new recipe to database if it doesn't exist"""
        recipe_id = str(recipe['id'])
        if recipe_id in self.recipes:
            print(f"Recipe {recipe['title']} (ID: {recipe_id}) already exists in database")
            return False
        
        recipe['collected_at'] = datetime.now().isoformat()
        self.recipes[recipe_id] = recipe
        return True
    
    def save_database(self):
        """Save current state of database"""
        self.update_metadata()
        try:
            # Create backup before saving
            if os.path.exists(self.DATABASE_FILE):
                backup_name = f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.rename(self.DATABASE_FILE, backup_name)
            
            with open(self.DATABASE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'recipes': list(self.recipes.values())
                }, f, indent=2)
            print(f"Successfully saved {len(self.recipes)} recipes to database")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def update_metadata(self):
        """Update database metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['total_recipes'] = len(self.recipes)
        self.metadata['cuisine_distribution'] = self._get_cuisine_distribution()
        self.metadata['diet_distribution'] = self._get_diet_distribution()
        self.metadata['dish_type_distribution'] = self._get_dish_type_distribution()
    
    def _get_cuisine_distribution(self) -> Dict[str, int]:
        distribution = {}
        for recipe in self.recipes.values():
            for cuisine in recipe['cuisineType']:
                distribution[cuisine] = distribution.get(cuisine, 0) + 1
        return distribution
    
    def _get_diet_distribution(self) -> Dict[str, int]:
        distribution = {}
        for recipe in self.recipes.values():
            for diet in recipe['diets']:
                distribution[diet] = distribution.get(diet, 0) + 1
        return distribution
    
    def _get_dish_type_distribution(self) -> Dict[str, int]:
        distribution = {}
        for recipe in self.recipes.values():
            for dish_type in recipe['dishTypes']:
                distribution[dish_type] = distribution.get(dish_type, 0) + 1
        return distribution

class SpoonacularCollector:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('SPOONACULAR_API_KEY')
        self.base_url = "https://api.spoonacular.com/recipes"
        self.recipes = []
        self.points_used = 0
        self.points_limit = 150
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure at least 0.5s between requests"""
        current_time = time.time()
        if current_time - self.last_request_time < 0.5:
            time.sleep(0.5 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting and point tracking"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params['apiKey'] = self.api_key
            
            print(f"\nMaking request to: {url}")
            response = requests.get(url, params=params)
            
            # Track points used from response headers
            points = round(float(response.headers.get('X-API-Quota-Used', '1')))
            self.points_used += points
            print(f"Points used in this request: {points}. Total points used: {self.points_used}")
            
            if response.status_code == 402:
                print("Daily points limit reached!")
                return None
                
            response.raise_for_status()
            
            # Debug: Print first recipe's nutrition data
            data = response.json()
            if data.get('recipes'):
                first_recipe = data['recipes'][0]
                print(f"\nSample nutrition data from first recipe:")
                print(f"Recipe: {first_recipe.get('title')}")
                print(f"Raw nutrition data: {first_recipe.get('nutrition', {})}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
    
    def fetch_recipes(self, target_count: int = 75) -> List[Dict]:
        """Fetch recipes using random recipe endpoint"""
        print(f"Starting recipe collection. Target: {target_count} recipes")
        
        while len(self.recipes) < target_count and self.points_used < self.points_limit:
            remaining_points = self.points_limit - self.points_used
            if remaining_points < 2:
                print("Insufficient points remaining")
                break
            
            batch_size = min(10, target_count - len(self.recipes), remaining_points // 2)
            
            params = {
                'number': batch_size,
                'addRecipeInformation': True,
                'includeNutrition': True,
                'fillIngredients': True
            }
            
            print(f"\nMaking API request with parameters: {params}")
            data = self._make_request('random', params)
            if not data:
                break
                
            recipes = data.get('recipes', [])
            print(f"Received {len(recipes)} recipes from API")
            
            for recipe in recipes:
                processed = self._process_recipe(recipe)
                if processed:
                    self.recipes.append(processed)
                    print(f"Successfully processed recipe: {processed['title']}")
                    print(f"Nutrition data: {processed['nutrition']}")
            
            print(f"Collected {len(self.recipes)} recipes so far")
            
        return self.recipes
    
    def _process_recipe(self, recipe: Dict) -> Optional[Dict]:
        """Process raw API response into standardized format"""
        try:
            # Get nutrition directly from API response's nutrients array
            nutrients = recipe.get('nutrition', {}).get('nutrients', [])
            nutrition = {
                "calories": next((nut['amount'] for nut in nutrients if nut['name'].lower() == 'calories'), 0.0),
                "protein": next((nut['amount'] for nut in nutrients if nut['name'].lower() == 'protein'), 0.0),
                "carbs": next((nut['amount'] for nut in nutrients if nut['name'].lower() == 'carbohydrates'), 0.0),
                "fat": next((nut['amount'] for nut in nutrients if nut['name'].lower() == 'fat'), 0.0)
            }
            
            # Log nutrition data for debugging
            print(f"\nProcessing recipe: {recipe['title']}")
            print(f"Raw nutrients: {nutrients}")
            print(f"Processed nutrition: {nutrition}")
            
            return {
                "id": recipe['id'],
                "title": recipe['title'],
                "readyInMinutes": recipe['readyInMinutes'],
                "servings": recipe['servings'],
                "pricePerServing": round(recipe['pricePerServing'], 2),
                "nutrition": nutrition,
                "ingredients": [
                    {
                        "name": ing['name'],
                        "amount": round(float(ing.get('amount', 0)), 2),
                        "unit": ing.get('unit', '')
                    } for ing in recipe.get('extendedIngredients', [])
                ],
                "cuisineType": recipe.get('cuisines', ['unknown']),
                "dishTypes": recipe.get('dishTypes', ['unknown']),
                "diets": recipe.get('diets', []),
                "instructions": recipe.get('instructions', ''),
                "image": recipe.get('image', ''),
                "sourceUrl": recipe.get('sourceUrl', ''),
                "healthScore": recipe.get('healthScore', 0),
                "cheap": recipe.get('cheap', False),
                "veryHealthy": recipe.get('veryHealthy', False),
                "veryPopular": recipe.get('veryPopular', False)
            }
        except KeyError as e:
            print(f"Error processing recipe: {e}")
            return None

def main():
    database = RecipeDatabase()
    collector = SpoonacularCollector()
    
    # Fetch new recipes
    new_recipes = collector.fetch_recipes()
    
    # Add new recipes to database
    recipes_added = 0
    for recipe in new_recipes:
        if database.add_recipe(recipe):
            recipes_added += 1
    
    # Save updated database
    database.save_database()
    
    # Print summary
    print("\nCollection Summary:")
    print(f"New recipes added: {recipes_added}")
    print(f"Total recipes in database: {len(database.recipes)}")
    print("\nCuisine Distribution:")
    print(json.dumps(database.metadata['cuisine_distribution'], indent=2))
    print("\nDiet Distribution:")
    print(json.dumps(database.metadata['diet_distribution'], indent=2))

if __name__ == "__main__":
    main()