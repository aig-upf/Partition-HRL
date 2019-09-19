import numpy as np

red = '\033[91m'
green = '\033[92m'
yellow = '\033[93m'
white = '\033[0m'
tab = '   '


def obs_equal(obs, other):
    if type(obs).__name__ == "ndarray" and type(other).__name__ == "ndarray":
        return np.array_equal(obs, other)

    elif type(obs).__name__ == "list" and type(other).__name__ == "list":
        return obs == other

    elif type(obs).__name__ == "tuple" and type(other).__name__ == "tuple":
        return obs == other

    elif type(obs).__name__ == "int" and type(other).__name__ == "int":
        return obs == other

    else:
        raise NotImplementedError("These observations cannot be compared. \n" +
                                  "First: " + str(type(obs).__name__) + "\n" +
                                  "Second: " + str(type(other).__name__))


def find_element_in_list(element, list_element):
    """
    find the element in the list.
    element can be of any type.
    :param element:
    :param list_element:
    :return:
    None if element is not in list_element
    the index of element in list_element
    """
    size_list = len(list_element)
    return next((i for e, i in zip(list_element, range(size_list)) if obs_equal(element, e)), None)


def make_tuple(image):
    return tuple(tuple(tuple(color) for color in lig) for lig in image)


def constrained_return_type(f):
    """
    Decorator to constrain the type of the output of a function.
    How to use it:

    @constrain_return_type
    def function(args)
    :param f:
    :return:
    """
    def decorated(*args, **kwargs):
        output = f(*args, **kwargs)
        class_annotation = f.__annotations__["return"]
        if not issubclass(type(output), class_annotation):
            raise TypeError("this class must return an object inheriting from " +
                            str(class_annotation.__name__) + " not " + str(type(output).__name__))

        return output

    return decorated


def constrained_type(object1, object2):
    """
    raises an error if object1 inherits from object2
    :param object1: the child class
    :param object2: the parent class
    :return:
    """
    if not issubclass(type(object1), object2):
        raise TypeError("this class must inherit from " +
                        object2.__name__ + " rather than " + type(object1).__name__)


def check_type(object1, object2):
    """
    check that object1 inherits from object2
    :param object1: the child class
    :param object2: the parent class
    :return:
    """
    return issubclass(type(object1), object2)
