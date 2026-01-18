
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

from config import Config
from models import MLP, MultiHeadMLP, SimpleCIFARCNN, MultiHeadCIFARCNN
from datasets import (
    build_permuted_mnist_tasks,
    build_rotated_mnist_tasks,
    build_split_mnist_tasks,
    build_split_mnist_ic_tasks,
    build_split_cifar10_tasks,
    build_split_cifar100_tasks,
)
from optimizers import ContinualMethod, FOPNGPreFisherMethod, SGDMethod, AdamMethod, OGDMethod, FOPNGMethod, FNGMethod, AVECollector, EWCMethod
from gradients import GTLCollector, GradientCollector
from fisher import DiagonalFisherEstimator, FullFisherEstimator, FisherEstimator, fisher_norm_distance
from utils import set_seed, evaluate
from logger import ExperimentLogger

def run_experiment(
    tasks: List[Tuple[DataLoader, DataLoader]],
    model: nn.Module,
    method: ContinualMethod,
    config: Config,
    multihead: bool = False,
    optimizer_class: type = None,
    task_names: Optional[List[str]] = None,
    dataset_name: str = "Unknown",
    logger: Optional[ExperimentLogger] = None
) -> Dict[int, List[float]]:
    """
    Run a continual learning experiment.
    
    Args:
        tasks: List of (train_loader, test_loader) tuples
        model: Neural network model
        method: Continual learning method
        config: Experiment configuration
        multihead: Whether model uses task-specific heads
        optimizer_class: Optimizer class (default: SGD for OGD, Adam for FOPNG)
        task_names: Optional names for each task
        dataset_name: Name of the dataset for logging
        logger: Optional ExperimentLogger for data collection
    
    Returns:
        Dictionary mapping task_id -> list of accuracies after each training task
    """
    model.to(config.device)
    
    # Default optimizer selection
    if optimizer_class is None:
        if isinstance(method, FOPNGMethod) or isinstance(method, AdamMethod) or isinstance(method, FOPNGPreFisherMethod):
            optimizer_class = torch.optim.Adam
        else:
            optimizer_class = torch.optim.SGD
    
    optimizer = optimizer_class(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    method.setup(model, config)
    results = defaultdict(list)
    train_results = defaultdict(list)
    num_tasks = len(tasks)
    
    # Setup logger
    if logger is None and config.log_dir:
        logger = ExperimentLogger(
            config.log_dir, 
            config.experiment_name, 
            config,
            project=config.wandb_project if config.use_wandb else 'fopng-experiments',
            entity=config.wandb_entity,
            tags=config.wandb_tags,
        )
    
    if logger:
        logger.start_experiment(method.name, dataset_name, task_names)
    
    prev_params = None
    param_distances = []
    for t in range(num_tasks):
        task_name = task_names[t] if task_names else f"Task {t}"
        print(f"\n{'='*60}")
        print(f"Training on {task_name}")
        print(f"{'='*60}")
        
        train_loader, _ = tasks[t]
        
        for epoch in range(config.epochs_per_task):
            loss, acc = method.train_epoch(
                model,
                optimizer,
                train_loader,
                criterion,
                config,
                t,
                multihead,
                progress_desc=f"Task {t} Epoch {epoch+1}/{config.epochs_per_task}"
            )
            
            print(f"Epoch {epoch+1}/{config.epochs_per_task} | "
                  f"Loss: {loss:.4f} | Acc: {acc*100:.2f}%")
            
            if logger:
                logger.log_epoch(
                    task_id=t,
                    epoch=epoch + 1,
                    train_loss=loss,
                    train_acc=acc
                )

        # Compute empirical Fisher
        # Track parameter change with true Fisher
        current_params = torch.cat([p.data.view(-1) for p in model.parameters()])
        if t > 0:
            train_loader_t, test_loader_t = tasks[t]
            
            fisher_dist_train = fisher_norm_distance(
                model, prev_params, current_params, train_loader_t, criterion, config.device
            )
            fisher_dist_test = fisher_norm_distance(
                model, prev_params, current_params, test_loader_t, criterion, config.device
            )
            l2_dist = torch.norm(current_params - prev_params).item()
            
            param_distances.append({
                'task': t,
                'fisher_distance_train': fisher_dist_train,
                'fisher_distance_test': fisher_dist_test,
                'l2_distance': l2_dist
            })
            print(f"  Parameter drift: L2={l2_dist:.4f}, Fisher(train)={fisher_dist_train:.4f}, Fisher(test)={fisher_dist_test:.4f}")
            
            # Log parameter distances to wandb (uses global step counter)
            from logger import log
            log({
                "param_drift/l2_distance": l2_dist,
                "param_drift/fisher_distance_train": fisher_dist_train,
                "param_drift/fisher_distance_test": fisher_dist_test,
                "trained_task": t,  # x-axis: which task we just finished training
            })
            
        prev_params = current_params.clone()
        
        # Evaluation on all tasks seen so far
        print("\nEvaluation:")
        for eval_t in range(num_tasks):
            train_loader_eval, test_loader_eval = tasks[eval_t]
            eval_name = task_names[eval_t] if task_names else f"Task {eval_t}"
            
            # Test accuracy
            test_loss, test_acc = evaluate(
                model, test_loader_eval, config.device, multihead, eval_t if multihead else None
            )
            train_loss, train_acc = evaluate(
                model, train_loader_eval, config.device, multihead, eval_t if multihead else None
            )
            
            results[eval_t].append(test_acc)
            train_results[eval_t].append(train_acc)
            
            print(f"  {eval_name}: Train={train_acc*100:.2f}% Test={test_acc*100:.2f}%")
            
            if logger:
                logger.log_eval(t, eval_t, test_loss, test_acc, train_loss, train_acc)
                # Store train accuracy - we'll need separate storage
        
        # After-task processing (gradient collection, etc.)
        method.after_task(model, train_loader, t, config, multihead)
    
    # Finalize logging
    if logger:
        # Note: logger.results is already populated during log_eval calls with both train/test data
        # Don't overwrite it with set_results - that would lose the train accuracy data
        logger.train_results = dict(train_results)  # Store for reference
        logger.end_experiment()
        logger.param_distances = param_distances
        
        if config.save_model:
            logger.save_model_checkpoint(model, "final")
        
        if config.save_plots:
            logger.create_all_plots()
        
        if config.save_raw_data:
            logger.save()
    
    return dict(results)

def compute_metrics(results: Dict[int, List[float]]) -> Dict[str, float]:
    """Compute continual learning metrics."""
    num_tasks = len(results)
    
    # Final accuracy (average across all tasks after all training)
    final_accs = [results[t][-1] for t in range(num_tasks)]
    avg_final_acc = np.mean(final_accs)
    
    # Forgetting (average accuracy drop for each task)
    forgetting = []
    for t in range(num_tasks - 1):
        max_acc = max(results[t])
        final_acc = results[t][-1]
        forgetting.append(max_acc - final_acc)
    avg_forgetting = np.mean(forgetting) if forgetting else 0.0
    
    return {
        'avg_final_accuracy': avg_final_acc,
        'avg_forgetting': avg_forgetting,
        'final_accuracies': final_accs
    }


def run_permuted_mnist(
    method_name: str = 'ogd',
    num_tasks: int = 5,
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """Run Permuted MNIST experiment."""
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Permuted MNIST ({num_tasks} tasks) — {method_name.upper()} ###")
    
    tasks = build_permuted_mnist_tasks(num_tasks, config.batch_size)
    model = MLP(input_dim=784, hidden_dim=100, num_classes=10)
    
    method = _create_method(method_name, **method_kwargs)
    
    # Setup logger
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"permuted_mnist_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=False,
        dataset_name="Permuted MNIST",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def run_rotated_mnist(
    method_name: str = 'ogd',
    angles: Tuple[float, ...] = (0, 10, 20, 30, 40),
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """Run Rotated MNIST experiment."""
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Rotated MNIST (angles={angles}) — {method_name.upper()} ###")
    
    tasks = build_rotated_mnist_tasks(angles, config.batch_size)
    model = MLP(input_dim=784, hidden_dim=100, num_classes=10)
    
    method = _create_method(method_name, **method_kwargs)
    task_names = [f"Rotation {a}°" for a in angles]
    
    # Setup logger
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"rotated_mnist_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=False,
        task_names=task_names,
        dataset_name="Rotated MNIST",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def run_split_mnist(
    method_name: str = 'ogd',
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """Run Split MNIST experiment."""
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Split MNIST (5 tasks × 2 digits) — {method_name.upper()} ###")
    
    tasks, digits_per_task = build_split_mnist_tasks(config.batch_size)
    model = MultiHeadMLP(
        input_dim=784,
        hidden_dim=100,
        num_heads=5,
        head_output_sizes=[2] * 5
    )
    
    method = _create_method(method_name, **method_kwargs)
    task_names = [f"Digits {d}" for d in digits_per_task]
    
    # Setup logger
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"split_mnist_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=True,
        task_names=task_names,
        dataset_name="Split MNIST",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def run_split_mnist_ic(
    method_name: str = 'ogd',
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """
    Run Split MNIST IC (Inference Class) experiment.
    
    Unlike run_split_mnist which uses 5 task-specific heads (2 outputs each),
    this uses a single shared 10-output MLP head. Tasks are still separated by digit classes.
    """
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Split MNIST IC (5 tasks × 2 digits, single 10-class head) — {method_name.upper()} ###")
    
    tasks, digits_per_task = build_split_mnist_ic_tasks(config.batch_size)
    model = MLP(input_dim=784, hidden_dim=100, num_classes=10)
    
    method = _create_method(method_name, **method_kwargs)
    task_names = [f"Digits {d}" for d in digits_per_task]
    
    # Setup logger
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"split_mnist_ic_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=False,
        task_names=task_names,
        dataset_name="Split MNIST IC",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def run_split_cifar10(
    method_name: str = 'ogd',
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """Run Split CIFAR-10 (5 tasks × 2 classes) experiment."""
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Split CIFAR-10 (5 tasks × 2 classes) — {method_name.upper()} ###")
    
    tasks, class_groups = build_split_cifar10_tasks(batch_size=config.batch_size)
    model = MultiHeadCIFARCNN(
        num_heads=len(class_groups),
        head_output_sizes=[len(group) for group in class_groups]
    )
    
    method = _create_method(method_name, **method_kwargs)
    task_names = [f"Classes {', '.join(map(str, group))}" for group in class_groups]
    
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"split_cifar10_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=True,
        task_names=task_names,
        dataset_name="Split CIFAR-10",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def run_split_cifar100(
    method_name: str = 'ogd',
    config: Optional[Config] = None,
    **method_kwargs
) -> Tuple[Dict[int, List[float]], Optional[ExperimentLogger]]:
    """Run Split CIFAR-100 (10 tasks × 10 classes) experiment."""
    config = config or Config()
    set_seed(config.seed)
    
    print(f"\n### Split CIFAR-100 (10 tasks × 10 classes) — {method_name.upper()} ###")
    
    tasks, class_groups = build_split_cifar100_tasks(batch_size=config.batch_size)
    model = MultiHeadCIFARCNN(
        num_heads=len(class_groups),
        head_output_sizes=[len(group) for group in class_groups]
    )
    
    method = _create_method(method_name, **method_kwargs)
    task_names = [f"Classes {', '.join(map(str, group))}" for group in class_groups]
    
    logger = None
    if config.log_dir:
        exp_name = config.experiment_name or f"split_cifar100_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = ExperimentLogger(config.log_dir, exp_name, config)
    
    results = run_experiment(
        tasks, model, method, config,
        multihead=True,
        task_names=task_names,
        dataset_name="Split CIFAR-100",
        logger=logger
    )
    
    metrics = compute_metrics(results)
    print(f"\nFinal avg accuracy: {metrics['avg_final_accuracy']*100:.2f}%")
    print(f"Avg forgetting: {metrics['avg_forgetting']*100:.2f}%")
    
    return results, logger


def _get_fisher_estimator(fisher_type: str = 'diagonal') -> FisherEstimator:
    """Create a Fisher estimator by type."""
    fisher_type = fisher_type.lower()
    if fisher_type == 'full':
        return FullFisherEstimator()
    else:
        return DiagonalFisherEstimator()


def _get_collector(collector_type: str = 'gtl') -> GradientCollector:
    """Create a gradient collector by type."""
    collector_type = collector_type.lower()
    if collector_type == 'ave':
        return AVECollector()
    else:
        return GTLCollector()


def _create_method(method_name: str, **kwargs) -> ContinualMethod:
    """Create a continual learning method by name."""
    method_name = method_name.lower()
    
    if method_name == 'sgd':
        return SGDMethod()
    elif method_name == 'adam':
        return AdamMethod()
    elif method_name == 'ogd':
        collector = _get_collector(kwargs.get('collector', 'gtl'))
        max_dirs = kwargs.get('max_directions', 2000)
        return OGDMethod(collector=collector, max_directions=max_dirs)
    elif method_name == 'ewc':
        fisher_est = _get_fisher_estimator(kwargs.get('fisher', 'diagonal'))
        return EWCMethod(fisher_estimator=fisher_est)
    elif method_name == 'fopng':
        fisher_est = _get_fisher_estimator(kwargs.get('fisher', 'diagonal'))
        collector = _get_collector(kwargs.get('collector', 'gtl'))
        max_dirs = kwargs.get('max_directions', 2000)
        return FOPNGMethod(
            fisher_estimator=fisher_est,
            collector=collector,
            max_directions=max_dirs
        )
    elif method_name == 'fopng_prefisher':
        fisher_est = _get_fisher_estimator(kwargs.get('fisher', 'diagonal'))
        collector = _get_collector(kwargs.get('collector', 'gtl'))
        max_dirs = kwargs.get('max_directions', 2000)
        return FOPNGPreFisherMethod(
            fisher_estimator=fisher_est,
            collector=collector,
            max_directions=max_dirs
        )
    elif method_name == 'fng':
        fisher_est = _get_fisher_estimator(kwargs.get('fisher', 'diagonal'))
        return FNGMethod(fisher_estimator=fisher_est)
    else:
        raise ValueError(f"Unknown method: {method_name}")

def make_exp_name(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [args.dataset, args.method]

    # Method-specific parts
    if args.method == "fopng":
        parts.append(args.fisher)
        parts.append(args.collector)
        parts.append(f"{args.max_directions}dirs")
        parts.append(f"lam{args.fopng_lambda_reg}")
    elif args.method == "fopng_prefisher":
        parts.append(args.fisher)
        parts.append(args.collector)
        parts.append(f"{args.max_directions}dirs")
        parts.append(f"lam{args.fopng_lambda_reg}")
    elif args.method == "fng":
        parts.append(args.fisher)
    elif args.method == "ewc":
        parts.append(args.fisher)
        parts.append(f"lambda{args.ewc_lambda}")
    elif args.method == "ogd":
        parts.append(args.collector)
        parts.append(f"{args.max_directions}dirs")

    # Dataset-specific parts
    if args.dataset == "permuted_mnist":
        parts.append(f"{args.num_tasks}tasks")
    elif args.dataset == "rotated_mnist":
        angstr = "-".join(str(a) for a in args.angles)
        parts.append(f"angles_{angstr}")
    elif args.dataset == "split_mnist":
        parts.append("5tasks")
    elif args.dataset == "split_mnist_ic":
        parts.append("5tasks")
    elif args.dataset == "split_cifar10":
        parts.append("5tasks")
    elif args.dataset == "split_cifar100":
        parts.append("10tasks")

    # Common parts
    parts.append(f"{args.epochs}epochs")
    parts.append(timestamp)

    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Continual Learning Experiments CLI")

    # ------------------------------
    # Core parameters
    # ------------------------------
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["permuted_mnist", "rotated_mnist", "split_mnist", "split_mnist_ic", "split_cifar10", "split_cifar100"])

    parser.add_argument("--method", type=str, required=True,
                        choices=["sgd", "adam", "ogd", "fopng", "fopng_prefisher", "fng", "ewc"])

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")

    # EWC-specific
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                        help="Regularization strength for EWC penalty")

    # --------------------------------
    # Dataset-specific
    # --------------------------------
    parser.add_argument("--num_tasks", type=int, default=5,
                        help="For permuted_mnist")

    parser.add_argument("--angles", type=float, nargs="+", default=[0, 10, 20, 30, 40],
                        help="For rotated_mnist")

    # --------------------------------
    # Method-specific
    # --------------------------------
    parser.add_argument("--collector", type=str, default="gtl",
                        choices=["gtl", "ave"])

    parser.add_argument("--fisher", type=str, default="diagonal",
                        choices=["diagonal", "full"])

    parser.add_argument("--max_directions", type=int, default=2000)
    parser.add_argument("--grads_per_task", type=int, default=200,
                        help="Number of gradient directions to collect per task for OGD/FOPNG")

    # FOPNG-specific
    parser.add_argument("--fopng_lambda_reg", type=float, default=0.0,
                        help="Regularization parameter for FOPNG")
    parser.add_argument("--fopng_new_fisher_weight", type=float, default=0.5,
                        help="Weight for new Fisher in weighted average: F_old = (1-w)*F_old + w*F_current")
    parser.add_argument("--use_empirical_fisher", action="store_true", default=False,
                        help="For FOPNG-PF: compute F*g on-the-fly during gradient collection instead of pre-multiplying by estimated Fisher")
    parser.add_argument("--fisher_batch_size", type=int, default=None,
                        help="If set, estimate Fisher from this batch size instead of the full training set")
    parser.add_argument("--first_task_lr", type=float, default=None,
                        help="If set, use this learning rate for the first task (task_id=0) instead of lr")
    parser.add_argument("--use_adam", action="store_true", default=False,
                        help="If set, use Adam optimizer for first task instead of SGD")
    parser.add_argument("--use_sgd", action="store_true", default=False,
                        help="If set, use SGD optimizer for first task instead of Adam")

    # --------------------------------
    # Logging / saving
    # --------------------------------
    parser.add_argument("--log_dir", type=str, default="./experiments")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--save_raw_data", action="store_true", default=True)
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="fopng",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity/team name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="Tags for wandb run")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()

    # ------------------------------
    # Build config
    # ------------------------------
    exp_name = make_exp_name(args)
    out_dir = Path(args.log_dir) / exp_name

    config = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_per_task=args.epochs,
        grads_per_task=args.grads_per_task,
        device=args.device,

        # Logging
        log_dir=str(out_dir),
        experiment_name=exp_name,
        save_model=args.save_model,
        save_plots=args.save_plots,
        save_raw_data=args.save_raw_data,
        
        # Wandb configuration
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=args.wandb_tags,
        use_wandb=not args.no_wandb,

        # FOPNG specific
        fopng_lambda_reg=args.fopng_lambda_reg,
        fopng_new_fisher_weight=args.fopng_new_fisher_weight,
        use_empirical_fisher=args.use_empirical_fisher,

        # Fisher
        fisher_batch_size=args.fisher_batch_size,

        # EWC
        ewc_lambda=args.ewc_lambda,

        # First task learning rate
        first_task_lr=args.first_task_lr,
        
        # Use Adam for first task
        use_adam=args.use_adam,
        
        # Use SGD for first task
        use_sgd=args.use_sgd,
    )

    # --------------------------------------------------------------------
    # Run the chosen experiment
    # --------------------------------------------------------------------
    if args.dataset == "permuted_mnist":
        run_permuted_mnist(
            args.method,
            num_tasks=args.num_tasks,
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )

    elif args.dataset == "rotated_mnist":
        run_rotated_mnist(
            args.method,
            angles=tuple(args.angles),
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )

    elif args.dataset == "split_mnist":
        run_split_mnist(
            args.method,
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )
    elif args.dataset == "split_mnist_ic":
        run_split_mnist_ic(
            args.method,
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )
    elif args.dataset == "split_cifar10":
        run_split_cifar10(
            args.method,
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )
    elif args.dataset == "split_cifar100":
        run_split_cifar100(
            args.method,
            config=config,
            collector=args.collector,
            fisher=args.fisher,
            max_directions=args.max_directions,
        )


if __name__ == "__main__":
    main()