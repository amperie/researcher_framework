"""All pipeline step node functions, discoverable by run_node.py."""
from graph.nodes.research import research_node
from graph.nodes.ideate import ideate_node
from graph.nodes.refine import refine_node
from graph.nodes.propose_experiments import propose_experiments_node
from graph.nodes.plan_implementation import plan_implementation_node
from graph.nodes.implement import implement_node
from graph.nodes.validate import validate_node
from graph.nodes.prepare_experiment import prepare_experiment_node
from graph.nodes.execute_experiment import execute_experiment_node
from graph.nodes.submit_experiment_jobs import submit_experiment_jobs_node
from graph.nodes.check_experiment_jobs import check_experiment_jobs_node
from graph.nodes.create_dataset import create_dataset_node
from graph.nodes.run_experiment import run_experiment_node
from graph.nodes.create_model import create_model_node
from graph.nodes.evaluate import evaluate_node
from graph.nodes.store_results import store_results_node
from graph.nodes.propose_next_steps import propose_next_steps_node

__all__ = [
    "research_node",
    "ideate_node",
    "refine_node",
    "propose_experiments_node",
    "plan_implementation_node",
    "implement_node",
    "validate_node",
    "prepare_experiment_node",
    "execute_experiment_node",
    "submit_experiment_jobs_node",
    "check_experiment_jobs_node",
    "create_dataset_node",
    "run_experiment_node",
    "create_model_node",
    "evaluate_node",
    "store_results_node",
    "propose_next_steps_node",
]

# Map step names (as declared in profile pipeline.steps) to node functions
STEP_REGISTRY: dict[str, object] = {
    "research": research_node,
    "ideate": ideate_node,
    "refine": refine_node,
    "propose_experiments": propose_experiments_node,
    "plan_implementation": plan_implementation_node,
    "implement": implement_node,
    "validate": validate_node,
    "prepare_experiment": prepare_experiment_node,
    "execute_experiment": execute_experiment_node,
    "submit_experiment_jobs": submit_experiment_jobs_node,
    "check_experiment_jobs": check_experiment_jobs_node,
    "create_dataset": create_dataset_node,
    "run_experiment": run_experiment_node,
    "create_model": create_model_node,
    "evaluate": evaluate_node,
    "store_results": store_results_node,
    "propose_next_steps": propose_next_steps_node,
}
