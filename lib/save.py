import os
import json
from typing import Any, Dict, Optional

import torch

# Public helper to move nested tensors to CPU for safe serialization
def to_cpu(obj: Any) -> Any:
	if isinstance(obj, torch.Tensor):
		return obj.detach().cpu()
	if isinstance(obj, dict):
		return {k: to_cpu(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		t = [to_cpu(v) for v in obj]
		return type(obj)(t) if isinstance(obj, tuple) else t
	return obj


def _experiment_params_to_config(params: Any, rnn: torch.nn.Module) -> Dict[str, Any]:
	"""Extract a JSON-serializable experiment config from params and model metadata.

	Avoids serializing model objects/transforms while preserving what we need
	to reproduce the run: hyperparameters, sizes, and library versions.
	"""
	# p can be a float or list in caller code; normalize to list for JSON
	p_value = getattr(params, "p", None)
	if isinstance(p_value, list):
		p_list = p_value
	elif p_value is None:
		p_list = None
	else:
		p_list = [p_value]

	# Pull sizes from the actual RNN to avoid relying on params shape bookkeeping
	in_features = getattr(rnn.W_in, "in_features", None)
	hidden_size = getattr(rnn, "hidden_size", None)
	out_features = getattr(rnn.W_out, "out_features", None)

	try:
		import torchvision  # type: ignore
		tv_version = getattr(torchvision, "__version__", None)
	except Exception:
		tv_version = None

	cfg = {
		# Task & data
		"dims": getattr(params, "dims", None),
		"input_type": getattr(params, "input_type", None),
		"input_size": getattr(params, "input_size", None),
		"p": p_list,
		"q": getattr(params, "q", None),
		"kappas": getattr(params, "kappas", None),

		# Model & dynamics
		"tuning_concentration": getattr(params, "tuning_concentration", None),
		"A": getattr(params, "A", None),
		"hidden_size": hidden_size,
		"rnn_input_size": in_features,
		"output_type": getattr(params, "output_type", None),
		"output_size": out_features,
		"dt": getattr(params, "dt", None),
		"tau": getattr(params, "tau", None),
		"eps1": getattr(params, "eps1", None),
		"C": getattr(params, "C", None),

		# Training
		"lr": getattr(params, "lr", None),
		"batch_size": getattr(params, "batch_size", None),
		"num_batches": getattr(params, "num_batches", None),
		"interactive": getattr(params, "interactive", None),

		# Testing
		"test_batch_size": getattr(params, "test_batch_size", None),

		# Device as string for info only
		"device": str(getattr(params, "device", "cpu")),

		# CNN metadata (identifiers only)
		"cnn_arch": ("convnext_base" if getattr(params, "input_type", None) == "feature" else None),
		# We can't safely import ConvNeXt_Base_Weights here; record a generic marker
		"cnn_weights": "DEFAULT" if getattr(params, "input_type", None) == "feature" else None,

		# Versions
		"torch_version": torch.__version__,
		"torchvision_version": tv_version,
	}
	return cfg


def save_experiment(
	experiment_output: Dict[str, Any],
	out_dir: str,
	*,
	save_test_batch: bool = False,
	save_cnn_features: bool = False,
) -> None:
	"""Save an experiment snapshot to out_dir.

	Artifacts:
	- config.json: JSON-serializable run config and sizes
	- rnn_state.pt: PyTorch state_dict for the RNN
	- losses.json: List[float] of training losses
	- test_batch.pt: Optional, CPU-copied dict of tensors
	- cnn_features_state.pt: Optional, feature extractor state_dict (rarely needed)
	"""
	os.makedirs(out_dir, exist_ok=True)

	params = experiment_output.get("params")
	rnn = experiment_output.get("rnn")
	losses = experiment_output.get("losses", [])
	test_batch = experiment_output.get("test_batch")

	if rnn is None:
		raise ValueError("experiment_output missing 'rnn'")
	if params is None:
		raise ValueError("experiment_output missing 'params'")

	# 1) Config
	cfg = _experiment_params_to_config(params, rnn)
	with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(cfg, f, indent=2)

	# 2) Model weights
	torch.save(rnn.state_dict(), os.path.join(out_dir, "rnn_state.pt"))

	# 3) Training curve
	with open(os.path.join(out_dir, "losses.json"), "w", encoding="utf-8") as f:
		json.dump([float(x) for x in losses], f)

	# 4) Optional: test batch
	if save_test_batch and test_batch is not None:
		torch.save(to_cpu(test_batch), os.path.join(out_dir, "test_batch.pt"))

	# 5) Optional: CNN features state (if a frozen extractor was used)
	if save_cnn_features and getattr(params, "input_type", None) == "feature":
		model = getattr(params, "model", None)
		if model is not None and hasattr(model, "features"):
			torch.save(model.features.state_dict(), os.path.join(out_dir, "cnn_features_state.pt"))


from types import SimpleNamespace


def params_from_config(cfg: Dict[str, Any], *, build_cnn: bool = False) -> Any:
	"""Recreate a params-like object from a saved config.

	If build_cnn is False (default), returns a lightweight object (SimpleNamespace)
	with the attributes needed by analysis utilities (e.g., output_type, device).
	If build_cnn is True, instantiates main.ExperimentParams which may construct
	the CNN (ConvNeXt) for feature inputs.
	"""
	device = torch.device(cfg.get("device", "cpu"))

	if not build_cnn:
		# Minimal object sufficient for analyze_test_batch and plotting
		return SimpleNamespace(
			dims=cfg.get("dims"),
			input_type=cfg.get("input_type"),
			input_size=cfg.get("input_size"),
			p=cfg.get("p"),
			q=cfg.get("q"),
			kappas=cfg.get("kappas"),
			hidden_size=cfg.get("hidden_size"),
			output_type=cfg.get("output_type"),
			tuning_concentration=cfg.get("tuning_concentration"),
			A=cfg.get("A"),
			dt=cfg.get("dt"),
			tau=cfg.get("tau"),
			eps1=cfg.get("eps1"),
			C=cfg.get("C"),
			lr=cfg.get("lr"),
			batch_size=cfg.get("batch_size"),
			num_batches=cfg.get("num_batches"),
			interactive=cfg.get("interactive", False),
			test_batch_size=cfg.get("test_batch_size"),
			device=device,
			model=None,
			preprocess=None,
			output_size=cfg.get("output_size"),
			eps2=(1.0 - float(cfg.get("eps1", 0.8)) ** 2) ** 0.5,
		)

	# Build full ExperimentParams (may build CNN for feature inputs)
	from main import ExperimentParams  # local import to avoid circular refs at module import

	p_val = cfg.get("p", [0.5])
	if p_val is None:
		p_val = [0.5]

	params = ExperimentParams(
		dims=cfg.get("dims", 1),
		input_type=cfg.get("input_type", "spatial"),
		input_size=cfg.get("input_size", 10),
		p=p_val,
		q=cfg.get("q", 0.0),
		kappas=cfg.get("kappas", [8]),
		hidden_size=cfg.get("hidden_size", 100),
		output_type=cfg.get("output_type", "angle"),
		tuning_concentration=cfg.get("tuning_concentration", 8.0),
		A=cfg.get("A", 1.0),
		dt=cfg.get("dt", 0.01),
		tau=cfg.get("tau", 0.25),
		eps1=cfg.get("eps1", 0.8),
		C=cfg.get("C", 0.1),
		lr=cfg.get("lr", 0.001),
		batch_size=cfg.get("batch_size", 200),
		num_batches=cfg.get("num_batches", 2000),
		interactive=cfg.get("interactive", False),
		test_batch_size=cfg.get("test_batch_size", 0),
	)
	# Set saved output_size if present
	if "output_size" in cfg:
		params.output_size = cfg["output_size"]
	return params


def load_experiment(
	out_dir: str,
	map_location: Optional[str] = "cpu",
	*,
	reconstruct_params: bool = True,
	build_cnn: bool = False,
) -> Dict[str, Any]:
	"""Load a previously saved experiment snapshot.

	Returns a dict with keys: config, rnn, losses, test_batch (optional).
	RNN is reconstructed using saved state shapes and dt/tau from config.
	"""
	from lib.rnn import RNN  # Local import to avoid circular deps

	cfg_path = os.path.join(out_dir, "config.json")
	state_path = os.path.join(out_dir, "rnn_state.pt")
	losses_path = os.path.join(out_dir, "losses.json")
	test_batch_path = os.path.join(out_dir, "test_batch.pt")

	if not os.path.isfile(cfg_path) or not os.path.isfile(state_path):
		raise FileNotFoundError("Missing config.json or rnn_state.pt in out_dir")

	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	# Load state dict first to infer sizes
	state_dict = torch.load(state_path, map_location=map_location)
	# Infer sizes from weight matrices
	try:
		w_in = state_dict["W_in.weight"]  # [hidden, input]
		w = state_dict["W.weight"]  # [hidden, hidden]
		w_out = state_dict["W_out.weight"]  # [output, hidden]
		input_size = w_in.shape[1]
		hidden_size = w.shape[0]
		output_size = w_out.shape[0]
	except KeyError as e:
		raise KeyError(f"State dict missing expected key: {e}")

	dt = cfg.get("dt")
	tau = cfg.get("tau")
	if dt is None or tau is None:
		raise ValueError("Config missing dt/tau required to rebuild RNN")

	rnn = RNN(input_size, hidden_size, output_size, dt, tau)
	rnn.load_state_dict(state_dict)
	if map_location:
		rnn.to(map_location)

	# Losses
	losses = []
	if os.path.isfile(losses_path):
		with open(losses_path, "r", encoding="utf-8") as f:
			losses = json.load(f)

	# Optional test batch
	test_batch = None
	if os.path.isfile(test_batch_path):
		test_batch = torch.load(test_batch_path, map_location=map_location)


	params_obj = None
	if reconstruct_params:
		params_obj = params_from_config(cfg, build_cnn=build_cnn)

	return {"config": cfg, "params": params_obj, "rnn": rnn, "losses": losses, "test_batch": test_batch}
