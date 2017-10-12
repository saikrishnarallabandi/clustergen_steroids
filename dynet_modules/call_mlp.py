from MLP import OneLayerMLP
import dynet as dy


m = dy.ParameterCollection()
# create an embedding table.
E = m.add_lookup_parameters((1000,10))
# create an MLP from 10 to 4 with a hidden layer of 20.
mlp = OneLayerMLP(m, 10, 20, 4, dy.rectify)

# use them together.
output = mlp(E[3])
# now save the model:
dy.save("basename",[mlp, E])
# now load:
m2 = dy.ParameterCollection()
mlp2, E2 = dy.load("basename", m2)
output2 = mlp2(E2[3])

import numpy
assert(numpy.array_equal(output2.npvalue(), output.npvalue()))
print "Done"
