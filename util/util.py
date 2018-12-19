##############################################################################################
#	Created by Fuyuan Lyu Tommy at 2018-12/18
#	MIT License
#   Contain utility functions
##############################################################################################


# If input is a single int, output is a two-element list and both equals to input
# If input is a list, return the same
def make_double(old):
	if isinstance(old,int):
		new = [old,old]
	elif isinstance(old,list):
		new = old
	else:
		print('Error!! Not right input type!!')
		return
	return new