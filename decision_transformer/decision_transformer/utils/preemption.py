import os
import pickle
import time
from typing import Any


class CheckpointTimer:
	"""
	Timer to keep track of when to checkpoint
	"""
	def __init__(self, checkpoint_every: int):
		self.checkpoint_every = checkpoint_every
		self.last_checkpoint = 0

	def should(self):
		return time.time() - self.last_checkpoint >= self.checkpoint_every

	def done(self):
		self.last_checkpoint = time.time()


class PreemptionManager:
	"""
	Manager to handle preemption in the training loop.
	"""
	def __init__(
			self, 
			checkpoint_dir: str, 
			checkpoint_every: int = 0, 
			checkpoint_timer: CheckpointTimer = None, 
			prefix: str = ''
		):
		self.checkpoint_dir = checkpoint_dir
		if checkpoint_timer is None:
			self.checkpoint_timer = CheckpointTimer(checkpoint_every)
		else:
			self.checkpoint_timer = checkpoint_timer
		self.prefix = prefix
		self.stored = dict()

	def _load_data(self, name: str):
		if self.checkpoint_dir is not None:
			path = os.path.join(self.checkpoint_dir, f'{self.prefix}_{name}.pkl')
			if os.path.exists(path):
				with open(path, 'rb') as file:
					print(f'Loaded {name}...')
					data = pickle.load(file)
				return data
		return None
	
	def load_if_exists(self, name: str, default_value: Any):
		data = self._load_data(name)
		if data is None:
			return default_value
		return data

	def load_torch(self, name: str, torch_class: Any, *args, **kwargs):
		state_dict = self._load_data(name)
		model = torch_class(*args, **kwargs)
		if state_dict is not None:
			model.load_state_dict(state_dict)
		return model
	
	def save(self, name: str, data: Any, now: bool = False):
		if now:
			if self.checkpoint_dir is not None:
				if not os.path.exists(self.checkpoint_dir):
					os.system(f"mkdir {self.checkpoint_dir}")
				path = os.path.join(self.checkpoint_dir, f'{self.prefix}_{name}.pkl')
				with open(path, 'wb') as path:
					pickle.dump(data, path)
		else:
			self.stored[name] = data

	def checkpoint(self):
		if self.checkpoint_timer.should():
			for key in self.stored:
				self.save(key, self.stored[key], now=True)
			self.checkpoint_timer.done()
			self.stored = dict()
