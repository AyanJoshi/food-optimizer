import numpy as np
from deap import base, creator, tools, algorithms
import random
import streamlit as st
from typing import List, Dict, Tuple, Optional
import requests
from dataclasses import dataclass
import pandas as pd
import os
import time
from requests.exceptions import RequestException
from dotenv import load_dotenv
from spoonacular_recipe_getter import RecipeDatabase
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# Load environment variables
load_dotenv()

def setup_deap_types():
    """Setup DEAP creator types if they don't exist"""
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize DEAP types
setup_deap_types()

@dataclass
class OptimizationResult:
    meal_plan: List[Dict]
    metrics: Dict[str, float]
    algorithm: str

class NSGAVariants:
    def __init__(self, budget_limit: float = None, daily_calories: float = None):
        self.toolbox = base.Toolbox()
        self.database = RecipeDatabase()
        self.budget_limit = budget_limit
        self.daily_calories = daily_calories 
        self._setup_base_ga()
        
    def _setup_base_ga(self):
        """Setup basic genetic algorithm parameters"""
        print("Setting up Parameters")
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                            self._generate_random_meal, n=9)
        print("Registered individuals")
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        print("Population has been setup")
        self.toolbox.register("mate", tools.cxTwoPoint)
        print("Crossover has been set")
        self.toolbox.register("evaluate", self._evaluate_meal_plan)
        print("Evaluation of initial population done")
        # Register the custom mutation operator
        self.toolbox.register("mutate", self.mutReplaceMeal, indpb=0.2)
        print("Custom mutation operator registered")
    
    def _generate_random_meal(self) -> Dict:
        """Generate a random meal from the local recipe database"""
        recipes = list(self.database.recipes.values()) if isinstance(self.database.recipes, dict) else []
        # print("\n\nGenerating Random Recipe\n\n")
        # print(f"self.database.recipes: {len(recipes)}")
        if len(recipes)>1   :
            return random.choice(recipes)
        else:
            st.error("No recipes found in the local database.")
            return {
                "title": "Database Empty - Placeholder Recipe",
                "readyInMinutes": 30,
                "pricePerServing": 500,
                "nutrition": {"nutrients": []},
                "extendedIngredients": [],
                "cuisines": ["unknown"]
            }
    
    def _evaluate_meal_plan(self, individual: List[Dict]) -> Tuple[float, float, float, float, float]:
        total_cost = sum(m.get('pricePerServing',0) for m in individual)/100.0
        total_prep_time = sum(m.get('readyInMinutes',0) for m in individual)
        nutritional_balance = self._calculate_nutritional_balance(individual)
        variety_score = self._calculate_variety_score(individual)
        
        daily_calories_list = []
        for day_idx in range(3):
            start = day_idx*3
            day_meals = individual[start:start+3]
            day_cal = sum(m.get('nutrition', {}).get('calories',0) for m in day_meals)
            daily_calories_list.append(day_cal)

        # Suppose min_daily_calories = 2000
        calorie_deviation = sum(max(0, self.daily_calories - dc) for dc in daily_calories_list)

        # If budget limit is set
        if self.budget_limit is not None and total_cost > self.budget_limit:
            return (float('inf'), float('inf'), float('inf'), float('inf'), float('inf'))

        # Return all objectives: cost, time, nutrition_balance, variety, calorie_deviation
        return (total_cost, total_prep_time, nutritional_balance, variety_score, calorie_deviation)

    
    def _calculate_nutritional_balance(self, meals: List[Dict]) -> float:
        """Calculate nutritional balance score based on macros from the new data structure."""
        daily_nutrients = {
            'calories': 0.0,
            'protein': 0.0,
            'carbs': 0.0,
            'fat': 0.0
        }
        
        # Now we assume each meal's 'nutrition' dict has keys: 'calories', 'protein', 'carbs', 'fat'
        for meal in meals:
            meal_nutrition = meal.get('nutrition', {})
            daily_nutrients['calories'] += meal_nutrition.get('calories', 0.0)
            daily_nutrients['protein'] += meal_nutrition.get('protein', 0.0)
            daily_nutrients['carbs']   += meal_nutrition.get('carbs', 0.0)
            daily_nutrients['fat']     += meal_nutrition.get('fat', 0.0)
        
        # Calculate deviation from ideal ratios (assuming protein, carbs, fat are non-zero)
        total_macros = daily_nutrients['protein'] + daily_nutrients['carbs'] + daily_nutrients['fat']
        
        if total_macros == 0:
            # If still zero, it means no macro data available; return inf as before
            return float('inf')
            
        ideal_ratios = {'protein': 0.3, 'carbs': 0.5, 'fat': 0.2}
        actual_ratios = {
            macro: daily_nutrients[macro] / total_macros 
            for macro in ['protein', 'carbs', 'fat']
        }
        
        return sum((ideal_ratios[macro] - actual_ratios[macro])**2 for macro in ideal_ratios)

    
    def _calculate_variety_score(self, meals: List[Dict]) -> float:
        """Calculate how diverse the meal plan is"""
        # Count unique ingredients
        ingredients = set()
        for meal in meals:
            meal_ingredients = {ing['name'].lower() for ing in meal.get('extendedIngredients', [])}
            ingredients.update(meal_ingredients)
        
        # Penalize for repeated cuisines
        cuisines = [meal['cuisineType'][0] if meal.get('cuisineType') and len(meal['cuisineType']) > 0 else 'unknown' for meal in meals]
        cuisine_penalty = len(meals) - len(set(cuisines))
        
        return -len(ingredients) + (cuisine_penalty * 5)

    def mutReplaceMeal(self, individual: List[Dict], indpb: float = 0.2) -> Tuple[List[Dict]]:
        """
        Mutate an individual by replacing meals with a certain probability.
        
        Args:
            individual (List[Dict]): The meal plan to mutate.
            indpb (float): Independent probability for each meal to be replaced.
        
        Returns:
            Tuple[List[Dict]]: A tuple containing the mutated individual.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                new_meal = self._generate_random_meal()
                individual[i] = new_meal
        return (individual,)

    def _track_convergence(self, population) -> dict:
        """Track min, max, and average fitness for current generation"""
        fits = [ind.fitness.values for ind in population]
        return {
            'min': np.min(fits, axis=0),  # Best fitness for each objective
            'avg': np.mean(fits, axis=0),  # Average fitness for each objective
            'max': np.max(fits, axis=0)    # Worst fitness for each objective
        }
    
    def plot_convergence(self, basic_history, sharing_history, clearing_history):
        """Plot convergence over generations for all variants"""
        plt.figure(figsize=(10, 6))
        generations = range(len(basic_history))
        
        # Plot first objective (total cost) convergence
        plt.plot([h['avg'][0] for h in basic_history], label='Basic NSGA-II')
        plt.plot([h['avg'][0] for h in sharing_history], label='Fitness Sharing')
        plt.plot([h['avg'][0] for h in clearing_history], label='Clearing')
        
        plt.xlabel('Generation')
        plt.ylabel('Average Cost (Lower is better)')
        plt.title('Convergence Comparison of NSGA-II Variants')
        plt.legend()
        plt.grid(True)
        plt.savefig('convergence_plot.png')
        plt.close()

    def plot_population_distribution(self, basic_pop, sharing_pop, clearing_pop):
        """Plot final population distribution for all variants"""
        plt.figure(figsize=(10, 6))
        
        # Extract cost and prep_time objectives for visualization
        def get_objectives(population):
            objectives = [self._evaluate_meal_plan(ind) for ind in population]
            costs = [obj[0] for obj in objectives]  # First objective is cost
            times = [obj[1] for obj in objectives]  # Second objective is prep time
            return costs, times
        
        # Plot each variant's population
        b_costs, b_times = get_objectives(basic_pop)
        s_costs, s_times = get_objectives(sharing_pop)
        c_costs, c_times = get_objectives(clearing_pop)
        
        plt.scatter(b_costs, b_times, label='Basic NSGA-II', alpha=0.6)
        plt.scatter(s_costs, s_times, label='Fitness Sharing', alpha=0.6)
        plt.scatter(c_costs, c_times, label='Clearing', alpha=0.6)
        
        plt.xlabel('Total Cost ($)')
        plt.ylabel('Prep Time (minutes)')
        plt.title('Population Distribution in Objective Space')
        plt.legend()
        plt.grid(True)
        plt.savefig('population_distribution.png')
        plt.close()

    # def run_basic_nsga2(self, population_size: int = 50, generations: int = 30) -> List[OptimizationResult]:
    #     """Run basic NSGA-II"""
    #     print("Entering run_basic_nsga2()")
        
    #     # Ensure that selection is registered (already done in setup if not)
    #     self.toolbox.register("select", tools.selNSGA2)
        
    #     print("Creating initial population")
    #     pop = self.toolbox.population(n=population_size)
        
    #     print("Evaluating initial population")
    #     # Evaluate the entire population
    #     for i, individual in enumerate(pop):
    #         fitness = self.toolbox.evaluate(individual)
    #         individual.fitness.values = fitness
    #         print(f"Individual {i}: Fitness {fitness}")
        
    #     print("Running eaMuPlusLambda()")
    #     final_pop, logbook = algorithms.eaMuPlusLambda(
    #         pop, self.toolbox,
    #         mu=population_size, lambda_=population_size,
    #         cxpb=0.7, mutpb=0.3,
    #         ngen=generations,
    #         verbose=False
    #     )
    #     print("Final population is done")
        
    #     return self._create_optimization_results(final_pop, "Basic NSGA-II")

    def plot_all_objectives_convergence(self, basic_history, sharing_history, clearing_history):
        """Plot convergence for all objectives"""
        objectives = ['Cost', 'Prep Time', 'Nutritional Balance', 'Variety Score', 'Calorie Deviation']
        n_obj = len(objectives)
        
        # Calculate number of rows and columns needed
        n_rows = (n_obj + 2) // 3  # This will give us 2 rows for 5 plots
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, obj_name in enumerate(objectives):
            ax = axes[i]
            ax.plot([h['avg'][i] for h in basic_history], label='Basic NSGA-II')
            ax.plot([h['avg'][i] for h in sharing_history], label='Fitness Sharing')
            ax.plot([h['avg'][i] for h in clearing_history], label='Clearing')
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'Average {obj_name}')
            ax.grid(True)
            ax.legend()
        
        # Remove the 6th (empty) subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('all_objectives_convergence.png')
        plt.close()
    
    def plot_parallel_coordinates(self, basic_pop, sharing_pop, clearing_pop):
        plt.figure(figsize=(12, 6))
        
        def plot_population(population, label, color_name):
            data = np.array([self._evaluate_meal_plan(ind) for ind in population])
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            denominator = np.where((max_vals - min_vals) > 0, max_vals - min_vals, 1)
            data_norm = (data - min_vals) / denominator
            
            for row in data_norm:
                plt.plot(range(5), row, alpha=0.3, color=color_name, label=label)
        
        # Plot each population once for the legend
        plot_population(basic_pop[:1], 'Basic NSGA-II', 'blue')
        plot_population(sharing_pop[:1], 'Fitness Sharing', 'orange')
        plot_population(clearing_pop[:1], 'Clearing', 'green')
        
        # Plot rest of the populations without adding to legend
        plot_population(basic_pop[1:], '_nolegend_', 'blue')
        plot_population(sharing_pop[1:], '_nolegend_', 'orange')
        plot_population(clearing_pop[1:], '_nolegend_', 'green')
        
        plt.xticks(range(5), ['Cost ($)', 'Prep Time (min)', 'Nutrition Balance', 'Variety Score', 'Total Calories (kcal)'])
        plt.ylabel('Normalized Objective Values')
        plt.grid(True)
        plt.title('Parallel Coordinates Plot of All Objectives')
        plt.legend(loc='upper right')
        plt.savefig('parallel_coordinates.png')
        plt.close()

    def plot_scatter_matrix(self, basic_pop, sharing_pop, clearing_pop):
        """Create scatter plot matrix for all objectives"""
        objectives = ['Cost', 'Time', 'Nutrition', 'Variety', 'Calories']
        n_obj = 5
        fig, axes = plt.subplots(n_obj, n_obj, figsize=(15, 15))
        
        # Get objectives for each population
        def get_all_objectives(population):
            return np.array([self._evaluate_meal_plan(ind) for ind in population])
        
        basic_obj = get_all_objectives(basic_pop)
        sharing_obj = get_all_objectives(sharing_pop)
        clearing_obj = get_all_objectives(clearing_pop)
        
        # Plot each pair of objectives
        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]
                if i != j:
                    ax.scatter(basic_obj[:, j], basic_obj[:, i], alpha=0.6, label='Basic', color='blue')
                    ax.scatter(sharing_obj[:, j], sharing_obj[:, i], alpha=0.6, label='Sharing', color='orange')
                    ax.scatter(clearing_obj[:, j], clearing_obj[:, i], alpha=0.6, label='Clearing', color='green')
                else:
                    ax.text(0.5, 0.5, objectives[i], transform=ax.transAxes, 
                        horizontalalignment='center', verticalalignment='center')
                if i == n_obj-1:
                    ax.set_xlabel(objectives[j])
                if j == 0:
                    ax.set_ylabel(objectives[i])
        
        plt.tight_layout()
        plt.savefig('scatter_matrix.png')
        plt.close()

    def run_basic_nsga2(self, population_size: int = 100, generations: int = 100) -> Tuple[List[OptimizationResult], List[Dict], List]:
        """Run basic NSGA-II"""
        history = []
        self.toolbox.register("select", tools.selNSGA2)
        
        pop = self.toolbox.population(n=population_size)
        
        # Initial evaluation
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)
        
        for gen in range(generations):
            offspring = algorithms.varOr(pop, self.toolbox, 
                                    lambda_=population_size, 
                                    cxpb=0.7, mutpb=0.3)
            
            # Evaluate offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)
            
            # Select next generation
            pop = self.toolbox.select(pop + offspring, population_size)
            
            # Track convergence
            history.append(self._track_convergence(pop))
        
        # Create optimization results from final population
        results = self._create_optimization_results(pop, "Basic NSGA-II")
        return results, history, pop
    
    # def run_fitness_sharing_nsga2(self, population_size: int = 50, generations: int = 30, 
    #                             sharing_radius: float = 0.2) -> List[OptimizationResult]:
    #     """Run NSGA-II with fitness sharing"""
    #     print("Entering fitness sharing nsga2")
    #     if sharing_radius <= 0:
    #         raise ValueError("sharing_radius must be greater than zero to prevent division by zero.")
    #     # Custom selection with fitness sharing
    #     def selNSGA2WithSharing(individuals, k):
    #         selected = tools.selNSGA2(individuals, k)
            
    #         # Apply fitness sharing
    #         for ind in selected:
    #             shared_fit = 0.0
    #             for other in selected:
    #                 if ind != other:
    #                     distance = self._calculate_meal_plan_distance(ind, other)
    #                     if distance < sharing_radius:
    #                         sh = 1.0 - (distance/sharing_radius)**2
    #                         shared_fit += sh
                
    #             if shared_fit > 0:
    #                 ind.fitness.values = tuple(f/(1 + shared_fit) for f in ind.fitness.values)
            
    #         return selected
        
    #     self.toolbox.register("select", selNSGA2WithSharing)
    #     print("Creating initial population for fitness sharing")
    #     pop = self.toolbox.population(n=population_size)
    #     print("Running eaMuPlusLambda() with fitness sharing")
    #     final_pop = algorithms.eaMuPlusLambda(
    #         pop, self.toolbox,
    #         mu=population_size, lambda_=population_size,
    #         cxpb=0.7, mutpb=0.3,
    #         ngen=generations,
    #         verbose=False
    #     )
    #     print("Fitness sharing is done")
    #     return self._create_optimization_results(final_pop[0], "NSGA-II with Fitness Sharing")
    
    # def run_clearing_nsga2(self, population_size: int = 50, generations: int = 30, 
    #                       clearing_radius: float = 0.1, capacity: int = 2) -> List[OptimizationResult]:
    #     """Run NSGA-II with clearing"""
    #     print("Entering run clearing nsga2")
        
    #     # Custom selection with clearing
    #     def selNSGA2WithClearing(individuals, k):
    #         selected = tools.selNSGA2(individuals, k)
            
    #         # Apply clearing
    #         cleared = []
    #         for ind in selected:
    #             niche_count = 0
    #             for winner in cleared:
    #                 distance = self._calculate_meal_plan_distance(ind, winner)
    #                 if distance < clearing_radius:
    #                     niche_count += 1
    #                     if niche_count >= capacity:
    #                         ind.fitness.values = (float('inf'),) * len(ind.fitness.values)
    #                         break
                
    #             if niche_count < capacity:
    #                 cleared.append(ind)
            
    #         return cleared
        
    #     self.toolbox.register("select", selNSGA2WithClearing)
        
    #     pop = self.toolbox.population(n=population_size)
    #     print("Running eaMuPlusLambda() with clearing")
    #     final_pop = algorithms.eaMuPlusLambda(
    #         pop, self.toolbox,
    #         mu=population_size, lambda_=population_size,
    #         cxpb=0.7, mutpb=0.3,
    #         ngen=generations,
    #         verbose=False
    #     )
    #     print("Clearing is done")
    #     return self._create_optimization_results(final_pop[0], "NSGA-II with Clearing")

    def run_fitness_sharing_nsga2(self, population_size: int = 100, generations: int = 100, 
                                sharing_radius: float = 0.2) -> Tuple[List[OptimizationResult], List[Dict], List]:
        """Run NSGA-II with fitness sharing"""
        history = []
    
        if sharing_radius <= 0:
            raise ValueError("sharing_radius must be greater than zero to prevent division by zero.")
        
        # Custom selection with fitness sharing
        def selNSGA2WithSharing(individuals, k):
            selected = tools.selNSGA2(individuals, k)
            
            # Calculate shared fitness
            shared_fitnesses = []
            for ind in selected:
                shared_fit = 1.0
                for other in selected:
                    if ind != other:
                        distance = self._calculate_meal_plan_distance(ind, other)
                        if distance < sharing_radius:
                            sh = 1.0 - (distance/sharing_radius)**2
                            shared_fit += sh
                
                # Create new individual with modified fitness
                new_ind = creator.Individual(ind[:])  # Create proper DEAP individual
                new_ind.fitness = creator.FitnessMulti()
                new_ind.fitness.values = tuple(f/shared_fit for f in ind.fitness.values)
                shared_fitnesses.append((ind, new_ind.fitness.values))
            
            # Sort based on shared fitness but return original individuals
            return [ind for ind, _ in sorted(shared_fitnesses, 
                                        key=lambda x: x[1])][:k]
        
        self.toolbox.register("select", selNSGA2WithSharing)
        pop = self.toolbox.population(n=population_size)
        
        # Initial evaluation
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)
        
        for gen in range(generations):
            offspring = algorithms.varOr(pop, self.toolbox, 
                                    lambda_=population_size, 
                                    cxpb=0.7, mutpb=0.3)
            
            # Evaluate offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)
            
            # Select next generation
            pop = self.toolbox.select(pop + offspring, population_size)
            
            # Track convergence
            history.append(self._track_convergence(pop))
        
        results = self._create_optimization_results(pop, "NSGA-II with Fitness Sharing")
        return results, history, pop

    def run_clearing_nsga2(self, population_size: int = 100, generations: int = 100, 
                      clearing_radius: float = 0.1, capacity: int = 2) -> Tuple[List[OptimizationResult], List[Dict], List]:
        """Run NSGA-II with clearing"""
        history = []
        
        # Custom selection with clearing
        def selNSGA2WithClearing(individuals, k):
            selected = tools.selNSGA2(individuals, k)
            
            # Apply clearing
            cleared = []
            for ind in selected:
                niche_count = 0
                for winner in cleared:
                    distance = self._calculate_meal_plan_distance(ind, winner)
                    if distance < clearing_radius:
                        niche_count += 1
                        if niche_count >= capacity:
                            ind.fitness.values = (float('inf'),) * len(ind.fitness.values)
                            break
                
                if niche_count < capacity:
                    cleared.append(ind)
            
            return cleared
        
        self.toolbox.register("select", selNSGA2WithClearing)
        pop = self.toolbox.population(n=population_size)
        
        # Initial evaluation
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)
        
        for gen in range(generations):
            offspring = algorithms.varOr(pop, self.toolbox, 
                                    lambda_=population_size, 
                                    cxpb=0.7, mutpb=0.3)
            
            # Evaluate offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind)
            
            # Select next generation
            pop = self.toolbox.select(pop + offspring, population_size)
            
            # Track convergence
            history.append(self._track_convergence(pop))
        
        results = self._create_optimization_results(pop, "NSGA-II with Clearing")
        return results, history, pop
    
    def _calculate_meal_plan_distance(self, plan1: List[Dict], plan2: List[Dict]) -> float:
        # Compare costs
        cost_diff = abs(sum(m.get('pricePerServing', 0) for m in plan1) - 
                        sum(m.get('pricePerServing', 0) for m in plan2))
        
        # Compare prep times
        prep_diff = abs(sum(m.get('readyInMinutes', 0) for m in plan1) - 
                        sum(m.get('readyInMinutes', 0) for m in plan2))
        
        # Compare ingredients overlap
        ingredients1 = set()
        ingredients2 = set()
        for meal in plan1:
            ingredients1.update(ing['name'].lower() for ing in meal.get('extendedIngredients', []))
        for meal in plan2:
            ingredients2.update(ing['name'].lower() for ing in meal.get('extendedIngredients', []))
        
        union_ingredients = ingredients1.union(ingredients2)
        if not union_ingredients:
            # If both meal plans have no ingredients, define ingredient_similarity as 1.0 or 0.0
            # depending on how you want to handle this scenario.
            ingredient_similarity = 1.0
        else:
            intersection_ingredients = ingredients1.intersection(ingredients2)
            ingredient_similarity = len(intersection_ingredients) / len(union_ingredients)
        
        # Normalize and combine
        # Ensure no division by zero in normalization
        return (cost_diff/1000 + prep_diff/500 + (1 - ingredient_similarity)) / 3

    
    def _create_optimization_results(self, population: List[List[Dict]], 
                                   algorithm_name: str) -> List[OptimizationResult]:
        """Convert population into OptimizationResults"""
        results = []
        for individual in tools.sortNondominated(population, len(population))[0]:
            metrics = {
                'total_cost': sum(meal.get('pricePerServing', 0) for meal in individual)/100.0,
                'total_prep_time': sum(meal.get('readyInMinutes', 0) for meal in individual),
                'nutritional_balance': self._calculate_nutritional_balance(individual),
                'variety_score': self._calculate_variety_score(individual)
            }
            results.append(OptimizationResult(individual, metrics, algorithm_name))
        return results
    
    #Stat getter
    def run_statistical_analysis(self, num_runs=5):  # Reduced runs since it takes time
        """Run multiple times and collect statistics"""
        stats = {
            'Basic NSGA-II': {'times': [], 'final_metrics': []},
            'Fitness Sharing': {'times': [], 'final_metrics': []},
            'Clearing': {'times': [], 'final_metrics': []}
        }
        
        for _ in range(num_runs):
            for alg, run_func in [
                ('Basic NSGA-II', self.run_basic_nsga2),
                ('Fitness Sharing', self.run_fitness_sharing_nsga2),
                ('Clearing', self.run_clearing_nsga2)
            ]:
                start_time = time.time()
                results, _, _ = run_func()
                end_time = time.time()
                stats[alg]['times'].append(end_time - start_time)
                # Get average metrics across population
                avg_metrics = {
                    'total_cost': np.mean([r.metrics['total_cost'] for r in results]),
                    'total_prep_time': np.mean([r.metrics['total_prep_time'] for r in results]),
                    'nutritional_balance': np.mean([r.metrics['nutritional_balance'] for r in results]),
                    'variety_score': np.mean([r.metrics['variety_score'] for r in results]),
                    'calorie_deviation': np.mean([self._evaluate_meal_plan(r.meal_plan)[4] for r in results])
                }
                stats[alg]['final_metrics'].append(avg_metrics)
        
        # Print statistics
        with open('algorithm_statistics.txt', 'w') as f:
            for alg in stats:
                times = stats[alg]['times']
                metrics_list = stats[alg]['final_metrics']
                
                f.write(f"\n{alg}:\n")
                f.write(f"Average Runtime: {np.mean(times):.2f}s ± {np.std(times):.2f}s\n")
                f.write("Final Metrics (mean ± std):\n")
                for metric in ['total_cost', 'total_prep_time', 'nutritional_balance', 'variety_score', 'calorie_deviation']:
                    values = [m[metric] for m in metrics_list]
                    f.write(f"  {metric}: {np.mean(values):.2f} ± {np.std(values):.2f}\n")
        
        # Just print a simple message in Streamlit
        st.write("Statistical analysis complete. Results saved to 'algorithm_statistics.txt'")
        
        return stats
    
    def plot_statistics(self, stats):
        """Plot statistical comparisons between algorithms"""
        metrics = ['total_cost', 'total_prep_time', 'nutritional_balance', 'variety_score', 'calorie_deviation']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = [
                [m[metric] for m in stats[alg]['final_metrics']]
                for alg in ['Basic NSGA-II', 'Fitness Sharing', 'Clearing']
            ]
            
            bp = ax.boxplot(data, labels=['Basic', 'Sharing', 'Clearing'], 
                        patch_artist=True)
            
            # Color boxes
            colors = ['lightblue', 'lightgreen', 'pink']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric} Distribution')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        # Remove the 6th subplot and replace with box plot legend
        ax = axes[5]
        ax.set_axis_off()
        
        # Create box plot explanation
        explanation = """
        Box Plot Guide:
        ○ Outlier point
        ┬ Maximum
        ┌┐ 75% (Q3)
        ─ Median
        └┘ 25% (Q1)
        ┴ Minimum
        """
        ax.text(0.1, 0.5, explanation, 
                fontfamily='monospace',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('statistics_comparison.png')
        plt.close()

####----------------------------------------------BREAK-------------------------------------------------------------####

def enhanced_streamlit_interface():
    st.title("Advanced Meal Planning Optimization System")
    
    # User inputs
    st.sidebar.header("Preferences")
    height = st.sidebar.number_input("Height (cm)", 150, 200, 170)
    weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)
    diet = st.sidebar.selectbox("Dietary Restriction", ["None", "Vegetarian", "Vegan", "Gluten-Free"])
    goal = st.sidebar.selectbox("Health Goal", ["Lose Weight", "Gain Muscle", "Maintain"])
    budget_limit = st.sidebar.number_input("Weekly Budget ($)",  # Label for the input
        min_value=10.0,       # Minimum allowed value
        max_value=1000.0,     # Maximum allowed value
        value=300.0,          # Default value
        step=10.0             # Increment step
    )
    daily_calories = st.sidebar.number_input("Daily Calorie Requirement (kcal)",
        min_value=500,    # a reasonable minimum
        max_value=5000,   # a reasonable maximum
        value=2000,       # a sensible default
        step=100          # increments of 100 calories
    )
    if st.button("Generate Meal Plans"):
        try:
            st.write("Generating meal plans. Please wait...")
            optimizer = NSGAVariants(budget_limit=budget_limit, daily_calories=daily_calories)
            
            # Run statistical analysis
            stats = optimizer.run_statistical_analysis()
            optimizer.plot_statistics(stats)

            # Run variants with history tracking - Now collecting three values
            basic_results, basic_history, basic_pop = optimizer.run_basic_nsga2(population_size=100, generations=100)
            sharing_results, sharing_history, sharing_pop = optimizer.run_fitness_sharing_nsga2(population_size=100, generations=100)
            clearing_results, clearing_history, clearing_pop = optimizer.run_clearing_nsga2(population_size=100, generations=100)
            
            # Generate all plots
            optimizer.plot_all_objectives_convergence(basic_history, sharing_history, clearing_history)
            optimizer.plot_parallel_coordinates(basic_pop, sharing_pop, clearing_pop)
            optimizer.plot_scatter_matrix(basic_pop, sharing_pop, clearing_pop)

            st.write("Plots have been saved locally")
            # Pick one representative solution from each variant (assuming at least one result per list)
            if not basic_results:
                st.error("No results returned from Basic NSGA-II.")
                return
            if not sharing_results:
                st.error("No results returned from NSGA-II with Fitness Sharing.")
                return
            if not clearing_results:
                st.error("No results returned from NSGA-II with Clearing.")
                return
            
            basic_result = basic_results[0]
            sharing_result = sharing_results[0]
            clearing_result = clearing_results[0]
            
            # Create tabs for each algorithm variant
            tabs = st.tabs([basic_result.algorithm, sharing_result.algorithm, clearing_result.algorithm])
            
            for tab, result in zip(tabs, [basic_result, sharing_result, clearing_result]):
                with tab:
                    st.subheader(f"3-Day Meal Plan ({result.algorithm})")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Cost", f"${result.metrics['total_cost']:.2f}")
                    with col2:
                        st.metric("Prep Time", f"{result.metrics['total_prep_time']} min")
                    with col3:
                        # If nutritional_balance is inf, consider formatting or explaining
                        st.metric("Nutrition Score", f"{result.metrics['nutritional_balance']:.2f}")
                    with col4:
                        st.metric("Variety Score", f"{result.metrics['variety_score']:.2f}")
                    
                    # # Display meals with error handling
                    # for i, meal in enumerate(result.meal_plan):
                    #     try:
                    #         st.write(f"### Day {i+1}: {meal['title']}")
                    #         cols = st.columns([1, 2])
                    #         with cols[0]:
                    #             image_url = meal.get('image', '')
                    #             if image_url:
                    #                 try:
                    #                     st.image(image_url, width=200)
                    #                 except Exception as img_error:
                    #                     st.write("Unable to load image")
                    #                     print(f"Image error: {img_error}")
                    #         with cols[1]:
                    #             # Safely access nested dictionary values
                    #             calories = meal.get('nutrition', {}).get('calories', 'N/A')
                    #             st.write(f"Calories per serving: {calories} kcal")
                    #             st.write(f"Prep Time: {meal.get('readyInMinutes', 'N/A')} minutes")
                    #             st.write(f"Cost per serving: ${meal.get('pricePerServing', 0)/100:.2f}")
                                
                    #             # Safely handle cuisine types
                    #             cuisines = meal.get('cuisineType', [])
                    #             if cuisines and isinstance(cuisines, list):
                    #                 st.write(f"Cuisine: {', '.join(cuisines)}")
                    #             else:
                    #                 st.write("Cuisine: unknown")
                                
                    #             source = meal.get('sourceUrl')
                    #             if source:
                    #                 st.write(f"[View Source Recipe]({source})")
                    #     except Exception as e:
                    #         st.error(f"Error displaying meal {i+1}: {str(e)}")
                    #         print(f"Detailed error for meal {i+1}: {e}")

                    for day_idx in range(3):
                        st.write(f"### Day {day_idx+1}")
                        for meal_idx in range(3):
                            meal = result.meal_plan[day_idx*3 + meal_idx]
                            try:
                                cols = st.columns([1, 2])
                                with cols[0]:
                                    image_url = meal.get('image', '')
                                    if image_url:
                                        try:
                                            st.image(image_url, width=200)
                                        except Exception as img_error:
                                            st.write("Unable to load image")
                                            print(f"Image error: {img_error}")
                                with cols[1]:
                                    # Safely access nested dictionary values
                                    calories = meal.get('nutrition', {}).get('calories', 'N/A')
                                    st.write(f"### {meal['title']}")
                                    st.write(f"Calories per serving: {calories} kcal")
                                    st.write(f"Prep Time: {meal.get('readyInMinutes', 'N/A')} minutes")
                                    st.write(f"Cost per serving: ${meal.get('pricePerServing', 0)/100:.2f}")
                                    
                                    # Safely handle cuisine types
                                    cuisines = meal.get('cuisineType', [])
                                    if cuisines and isinstance(cuisines, list):
                                        st.write(f"Cuisine: {', '.join(cuisines)}")
                                    else:
                                        st.write("Cuisine: unknown")
                                    
                                    source = meal.get('sourceUrl')
                                    if source:
                                        st.write(f"[View Source Recipe]({source})")
                            except Exception as e:
                                st.error(f"Error displaying meal {i+1}: {str(e)}")
                                print(f"Detailed error for meal {i+1}: {e}")

                            
        except Exception as e:
            print(e)
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    enhanced_streamlit_interface()