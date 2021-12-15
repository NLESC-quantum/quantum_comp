# Period Finding Algorithm for Pulsar Detection

We explore here the possiblity to use the period finding algorithm to detect the periodicity of a pulsar. To this end we have developped a simple toy model of a  puslar that provides a simplified pulsar signal.

As explained in the notebook, the Shor's algorithm requires the function under examination to not only be periodic but also that the function takes unique values within a perdiod. This is a rather stringent requirement. We can however circumvent that limitation by integrating the pulsar signal over multiple channels to obtain an injective signal with the period (see details in the notebook). 

However even with this trick the signal requires to be pretty much noiseless for the approach to lead acceptable results. It is consequently not an ideal solution for leveraging quantum computing for radio astronomy.
