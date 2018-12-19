##############################################################################################
#	Created by Fuyuan Lyu Tommy at 2018-12/18
#	MIT License
##############################################################################################

import net.nin as nin
import net.googlenet as googlenet

if __name__ == '__main__':
	mynin = nin.NIN(32,3)
	mynin.calculate_all()
	mynin.write_csv()
	#mygooglenet = googlenet.GoogLeNet(32,3)
	mygooglenet = googlenet.GoogLeNet(224,3)
	mygooglenet.calculate_all()
	mygooglenet.write_csv()









