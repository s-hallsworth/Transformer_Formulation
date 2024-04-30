"""
Test Transformer with toy optimal control problem
"""
import Pyomo_Toy.transformer_test as transformer_test
import transformer_intermediate_results
import toy_problem_setup

# Create test transformer
test_transformer = transformer_test.TestTransformer()

# Run tests
test_transformer.test_pyomo_input(toy_problem_setup.model, "input_var",transformer_intermediate_results.input)
#test_transformer.test_layer_norm(toy_problem_test.model, "toy_config.json", T=11, transformer_output=transformer_intermediate_results.layer_norm_output)