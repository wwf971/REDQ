from __future__ import annotations
def str2bool(arg_value):
	if isinstance(arg_value, bool):
		return arg_value
	if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
class CustomArgumentParser(argparse.ArgumentParser):
	def add_bool_arg(self, arg_name: str, default: bool=False, help=None):
		if arg_name.startswith("--"):
			arg_name.lstrip("-")
		if default:
			self.add_argument('--' + arg_name, nargs="?", const=True, type=str2bool, default=True, help=help)
		else:
			self.add_argument('--' + arg_name, nargs="?", const=True, type=str2bool, default=False, help=help)
def get_parser() -> CustomArgumentParser:
	return CustomArgumentParser()