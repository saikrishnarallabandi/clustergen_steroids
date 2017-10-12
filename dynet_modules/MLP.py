import dynet as dy

# Example of a user-defined saveable type.
class OneLayerMLP(object):
  def __init__(self, model, num_input, num_hidden, num_out, act=dy.tanh):
    self.W1 = model.add_parameters((num_hidden, num_input))
    self.W2 = model.add_parameters((num_out, num_hidden))
    self.b1 = model.add_parameters((num_hidden))
    self.b2 = model.add_parameters((num_out))
    self.act = act
    self.spec = (num_input, num_hidden, num_out, act)

  def __call__(self, input, output, classification_flag):
    W1 = dy.parameter(self.W1)
    W2 = dy.parameter(self.W2)
    b1 = dy.parameter(self.b1)
    b2 = dy.parameter(self.b2)
    g = self.act
    if classification_flag == 1:
       return dy.softmax(W2*g(W1*input + b1)+b2)
    else:
       prediction = W2*g(W1*input + b1)+b2
       losses = output - prediction
       return dy.l2_norm(losses)

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, num_hidden, num_out, act = spec
    return OneLayerMLP(model, num_input, num_hidden, num_out, act)



