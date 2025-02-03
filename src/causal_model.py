from dowhy import CausalModel

def estimate_causal_effect(data, causal_graph, treatment, outcome):
    model = CausalModel(data=data, treatment=treatment, outcome=outcome, graph=causal_graph)
    identified_estimand = model.identify_effect()
    return model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
