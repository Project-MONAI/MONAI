
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from functools import wraps
import monai

export = monai.utils.export("monai.utils")


@export
def timing(func):
    """
    This simple timing function decorator prints to stdout/logfile (it uses printFlush) how many seconds a call to the
    original function took to execute, as well as the name before and after the call.
    """

    @wraps(func)
    def timingwrap(*args, **kwargs):
        print(func.__name__, flush=True)
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "dT (s) =", (end - start), flush=True)
        return res

    return timingwrap


@export
class RestartGenerator:
    """
    Wraps a generator callable which will be called whenever this class is iterated and its result returned. This is
    used to create an iterator which can start iteration over the given generator multiple times.
    """

    def __init__(self, createGen):
        self.createGen = createGen

    def __iter__(self):
        return self.createGen()


@export
class MethodReplacer(object):
    """
    Base class for method decorators which can be used to replace methods pass to replaceMethod() with wrapped versions.
    """

    replaceListName = "__replacemethods__"

    def __init__(self, meth):
        self.meth = meth

    def replaceMethod(self, meth):
        """Return a new method to replace `meth` in the instantiated object, or `meth` to do nothing."""
        return meth

    def __set_name__(self, owner, name):
        """
        Add the (name,self.replaceMethod) pair to the list named by replaceListName in `owner`, creating the list and
        replacing the constructor of `owner` if necessary. The replaced constructor will call the old one then do the
        replacing operation of substituting, for each (name,self.replaceMethod) pair, the named method with the returned
        value from self.replaceMethod.
        """
        entry = (name, owner, self.replaceMethod)

        if not hasattr(owner, self.replaceListName):
            oldinit = owner.__init__

            # replace the constructor with a new one which calls the old then replaces methods
            @wraps(oldinit)
            def newinit(_self, *args, **kwargs):
                oldinit(_self, *args, **kwargs)

                # replace each listed method of this newly constructed object
                for m, owner, replacer in getattr(_self, self.replaceListName):
                    if isinstance(_self, owner):
                        meth = getattr(_self, m)
                        newmeth = replacer(meth)
                        setattr(_self, m, newmeth)

            setattr(owner, "__init__", newinit)
            setattr(owner, self.replaceListName, [entry])
        else:
            namelist = getattr(owner, self.replaceListName)

            if not any(nl[0] == name for nl in namelist):
                namelist.append(entry)

        setattr(owner, name, self.meth)
