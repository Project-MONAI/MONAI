# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps

__all__ = ["RestartGenerator", "MethodReplacer"]


class RestartGenerator:
    """
    Wraps a generator callable which will be called whenever this class is iterated and its result returned. This is
    used to create an iterator which can start iteration over the given generator multiple times.
    """

    def __init__(self, create_gen) -> None:
        self.create_gen = create_gen

    def __iter__(self):
        return self.create_gen()


class MethodReplacer:
    """
    Base class for method decorators which can be used to replace methods pass to replace_method() with wrapped versions.
    """

    replace_list_name = "__replacemethods__"

    def __init__(self, meth) -> None:
        self.meth = meth

    def replace_method(self, meth):
        """
        Return a new method to replace `meth` in the instantiated object, or `meth` to do nothing.
        """
        return meth

    def __set_name__(self, owner, name):
        """
        Add the (name,self.replace_method) pair to the list named by replace_list_name in `owner`, creating the list and
        replacing the constructor of `owner` if necessary. The replaced constructor will call the old one then do the
        replacing operation of substituting, for each (name,self.replace_method) pair, the named method with the returned
        value from self.replace_method.
        """
        entry = (name, owner, self.replace_method)

        if not hasattr(owner, self.replace_list_name):
            oldinit = owner.__init__

            # replace the constructor with a new one which calls the old then replaces methods
            @wraps(oldinit)
            def newinit(_self, *args, **kwargs):
                oldinit(_self, *args, **kwargs)

                # replace each listed method of this newly constructed object
                for m, owner, replacer in getattr(_self, self.replace_list_name):
                    if isinstance(_self, owner):
                        meth = getattr(_self, m)
                        newmeth = replacer(meth)
                        setattr(_self, m, newmeth)

            owner.__init__ = newinit
            setattr(owner, self.replace_list_name, [entry])
        else:
            namelist = getattr(owner, self.replace_list_name)

            if not any(nl[0] == name for nl in namelist):
                namelist.append(entry)

        setattr(owner, name, self.meth)
