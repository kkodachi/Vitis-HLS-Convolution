# Vivado Issues

We attempted to synthesize overnight multiple times at on my home computer an in the lab. However synthesis crashed my home desktop and someone interfered with synthesis in the lab.

As a result we were forced to prune a few layers so that synthesis could be done on my home computer. squeezenetsynthesis1.png and squeezenetsynthesis2.png show that we were successfully able to synthesize after reducing the model size slightly. 

However moving onto Vivido we got more issues. The generating bitstream step reached the memory limit of both my home computer and lab computer as seen in both pair of screen shots. We were able to successfully generate the tcl file however task 4 was not possible for us due to hardware limits. Not sure how anyone was able to get past this considering we even tried reading the weights in instead of storing them on chip. However, the result was the same and we could not get past the memory limits for this.