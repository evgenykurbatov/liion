import numpy as np

class VarDecoder:
    def __init__(self):
        self.plain_setter_funcs = []
        self.plain_setter_names = []
        self.plain_setter_vals = []
        self.decoding_setter_names = []
        self.decoding_setter_funcs = []
        self.decoding_setter_lower = []
        self.decoding_setter_upper = []
        self.decoding_setter_sizes = []

    # Plain setter: sets functor(context).name to value
    def add_plain_setter(self, name, value, functor):
        self.plain_setter_names.append(name)
        self.plain_setter_funcs.append(functor)
        self.plain_setter_vals.append(value)

    # Decoding setter: sets functor(context).name to the next value(s) in the individual.
    # The value is:
    #   size = 0: just float
    #   size > 0: a slice of the individual
    # Functor takes either (context, Float) or (context, list[Float]) depending on the size.
    def add_decoding_setter(self, name, lower_bound, upper_bound, functor, size = 0):
        self.decoding_setter_names.append(name)
        self.decoding_setter_funcs.append(functor)
        self.decoding_setter_lower.append(lower_bound)
        self.decoding_setter_upper.append(upper_bound)
        self.decoding_setter_sizes.append(size)

    # Decodes the individual into the context.
    # Context can be anything, individual is an indexable of floats of a correct length.
    def decode(self, individual, context):
        for setter_idx in range(len(self.plain_setter_names)):
            obj = self.plain_setter_funcs[setter_idx](context)
            name = self.plain_setter_names[setter_idx]
            setattr(obj, name, self.plain_setter_vals[setter_idx])

        index = 0
        for setter_idx in range(len(self.decoding_setter_names)):
            obj = self.decoding_setter_funcs[setter_idx](context)
            name = self.decoding_setter_names[setter_idx]
            my_size = self.decoding_setter_sizes[setter_idx]
            if my_size == 0:
                # just a float
                setattr(obj, name, individual[index])
                index += 1
            else:
                setattr(obj, name, individual[index : index + my_size])
                index += my_size

    # Returns a tuple of (list of all lower bounds, list of all upper bounds).
    def bounds(self):
        lower = []
        upper = []
        for setter_idx in range(len(self.decoding_setter_names)):
            my_size = max(1, self.decoding_setter_sizes[setter_idx])
            for _ in range(my_size):
                lower.append(self.decoding_setter_lower[setter_idx])
                upper.append(self.decoding_setter_upper[setter_idx])
        return lower, upper

    # Samples a random individual uniformly.
    def random_individual(self):
        result = []
        for setter_idx in range(len(self.decoding_setter_names)):
            my_size = max(1, self.decoding_setter_sizes[setter_idx])
            lower = self.decoding_setter_lower[setter_idx]
            upper = self.decoding_setter_upper[setter_idx]
            for _ in range(my_size):
                result.append(np.random.uniform(low = lower, high = upper))
        return result

    # Dumps the individual to the console
    def print(self, individual):
        # print individual-dependent values
        index = 0
        for setter_idx in range(len(self.decoding_setter_names)):
            my_size = self.decoding_setter_sizes[setter_idx]
            name = self.decoding_setter_names[setter_idx]
            if my_size == 0:
                print(name + " = " + str(individual[index]))
                index += 1
            else:
                slice = individual[index : index + my_size]
                print(name + " = " + " ".join(str(v) for v in slice))
                index += my_size
        # print hardcoded values
        for setter_idx in range(len(self.plain_setter_names)):
            name = self.plain_setter_names[setter_idx]
            print(name + " = " + str(self.plain_setter_vals[setter_idx]) + " (synthesized)")
