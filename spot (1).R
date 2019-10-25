# set working directory to your files location
setwd("~/EvoMan_assignment")
# Or if using RStudio use Session->Set Working Directory

# Installation:
install.packages("devtools")
library("devtools")
install_github("bartzbeielstein/SPOT")

library("SPOT")
scriptLocation <- "spot_algorithm_A.py"
evolution <- wrapSystemCommand(paste("python", scriptLocation))
spot(x=NULL, evolution, c(0, 0), c(1, 1))

# you can use f in the GUI to play around with your algorithm
# install_github("frehbach/spotGUI")

# library("spotGUI")
# runSpotGUI()



