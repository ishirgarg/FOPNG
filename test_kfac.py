"""Test KFAC Fisher Estimator implementation."""
import torch
from torch.utils.data import TensorDataset, DataLoader
from fisher import KFACFisherEstimator
from models import MLP

def test_kfac():
    """Test KFAC estimation and inverse computation."""
    print("Testing KFAC Fisher Estimator...")
    
    # Create model
    model = MLP(784, 100, 10)
    estimator = KFACFisherEstimator()
    
    # Dummy data
    X = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=10)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Estimate
    print("  Estimating Fisher factors...")
    factors = estimator.estimate(model, loader, criterion, 'cpu')
    
    # Verify factors exist for all layers
    print("  Verifying factor shapes...")
    assert 'fc1' in factors, "fc1 not found in factors"
    assert 'fc2' in factors, "fc2 not found in factors"
    assert 'fc_out' in factors, "fc_out not found in factors"
    
    # Check shapes
    assert factors['fc1']['A'].shape == (785, 785), f"fc1 A shape wrong: {factors['fc1']['A'].shape}"  # 784 + 1 bias
    assert factors['fc1']['G'].shape == (100, 100), f"fc1 G shape wrong: {factors['fc1']['G'].shape}"
    
    assert factors['fc2']['A'].shape == (101, 101), f"fc2 A shape wrong: {factors['fc2']['A'].shape}"  # 100 + 1 bias
    assert factors['fc2']['G'].shape == (100, 100), f"fc2 G shape wrong: {factors['fc2']['G'].shape}"
    
    assert factors['fc_out']['A'].shape == (101, 101), f"fc_out A shape wrong: {factors['fc_out']['A'].shape}"
    assert factors['fc_out']['G'].shape == (10, 10), f"fc_out G shape wrong: {factors['fc_out']['G'].shape}"
    
    print("  ✓ All shapes correct")
    
    # Check symmetry
    print("  Checking symmetry...")
    for layer_name in ['fc1', 'fc2', 'fc_out']:
        A = factors[layer_name]['A']
        G = factors[layer_name]['G']
        assert torch.allclose(A, A.t(), atol=1e-5), f"{layer_name} A not symmetric"
        assert torch.allclose(G, G.t(), atol=1e-5), f"{layer_name} G not symmetric"
    print("  ✓ All matrices symmetric")
    
    # Check positive semi-definiteness (all eigenvalues >= 0, allowing for numerical error)
    print("  Checking positive semi-definiteness...")
    for layer_name in ['fc1', 'fc2', 'fc_out']:
        A = factors[layer_name]['A']
        G = factors[layer_name]['G']
        eig_A = torch.linalg.eigvalsh(A)
        eig_G = torch.linalg.eigvalsh(G)
        min_eig_A = eig_A.min().item()
        min_eig_G = eig_G.min().item()
        # Allow small negative eigenvalues due to numerical precision
        assert min_eig_A >= -1e-4, f"{layer_name} A has significantly negative eigenvalues: {min_eig_A}"
        assert min_eig_G >= -1e-4, f"{layer_name} G has significantly negative eigenvalues: {min_eig_G}"
    print("  ✓ All matrices positive semi-definite (within numerical precision)")
    
    # Check inverse computation
    print("  Testing inverse computation...")
    inv_factors = estimator.get_inverse_factors(damping=1e-3)
    
    for layer_name in ['fc1', 'fc2', 'fc_out']:
        A_inv, G_inv = inv_factors[layer_name]
        A = factors[layer_name]['A']
        G = factors[layer_name]['G']
        
        # Check that A_inv @ A ≈ I (with damping, won't be exact)
        # We'll check with the damped version
        damping = 1e-3
        sqrt_damping = torch.sqrt(torch.tensor(damping))
        pi = torch.sqrt((torch.trace(A) / A.size(0)) / (torch.trace(G) / G.size(0) + 1e-8))
        A_damped = A + pi * sqrt_damping * torch.eye(A.size(0))
        
        identity_A = A_inv @ A_damped
        assert torch.allclose(identity_A, torch.eye(A.size(0)), atol=1e-2), \
            f"{layer_name} A^{{-1}} A != I"
    
    print("  ✓ Inverse computation correct")
    
    # Test apply_inverse
    print("  Testing apply_inverse...")
    grad_dict = {
        'fc1': torch.randn(100, 784),
        'fc2': torch.randn(100, 100),
        'fc_out': torch.randn(10, 100),
    }
    
    precond_dict = estimator.apply_inverse(grad_dict, damping=1e-3)
    
    # Check that all layers are preconditioned
    assert 'fc1' in precond_dict
    assert 'fc2' in precond_dict
    assert 'fc_out' in precond_dict
    
    # Check shapes are preserved
    assert precond_dict['fc1'].shape == grad_dict['fc1'].shape
    assert precond_dict['fc2'].shape == grad_dict['fc2'].shape
    assert precond_dict['fc_out'].shape == grad_dict['fc_out'].shape
    
    print("  ✓ apply_inverse works correctly")
    
    print("\n✓✓✓ All KFAC tests passed! ✓✓✓\n")

if __name__ == '__main__':
    test_kfac()

