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

import monai

"""
How to use the adaptor function

The key to using 'adaptor' lies in understanding the function that want to
adapt. The 'inputs' and 'outputs' parameters take either strings, lists/tuples
of strings or a dictionary mapping strings, depending on call signature of the
function being called.

The adaptor function is written to minimise the cognitive load on the caller.
There should be a minimal number of cases where the caller has to set anything
on the input parameter, and for functions that return a single value, it is
only necessary to name the dictionary keyword to which that value is assigned.

Use of `outputs`

`outputs` can take either a string, a list/tuple of string or a dict of string
to string, depending on what the transform being adapted returns:
. If the transform returns a single argument, then outputs can be supplied a
  string that indicates what key to assign the return value to in the
  dictionary
. If the transform returns a list/tuple of values, then outputs can be supplied
  a list/tuple of the same length. The strings in outputs map the return value
  at the corresponding position to a key in the dictionary
. If the transform returns a dictionary of values, then outputs must be supplied
  a dictionary that maps keys in the function's return dictionary to the
  dictionary being passed between functions
  
Note, the caller is free to use a more complex way of specifying the outputs
parameter than is required. The following are synonymous and will be treated
identically:
```
   # single argument
   adaptor(MyTransform(), 'image')
   adaptor(MyTransform(), ['image'])
   adaptor(MyTransform(), {'image': 'image'})
   
   # multiple arguments
   adaptor(MyTransform(), ['image', 'label'])
   adaptor(MyTransform(), {'image': 'image', 'label': 'label'})
```

Use of `inputs`

`inputs` can usually be omitted when using `adaptor`. It is only required when a
the function's parameter names do not match the names in the dictionary that is
used to chain transform calls.

```
class MyTransform1:
    ...
    def __call__(image):
        return '''do stuff to image'''

class MyTransform2:
    ...
    def __call__(img):
        return '''do stuff to image'''

d = {'image': i}

Compose([
    adaptor(MyTransform1(), 'image'),
    adaptor(MyTransform2(), 'image', {'img':'image'})
])
```

"""
@monai.utils.export('monai.data.transforms')
def adaptor(function, outputs, inputs):

    def check_signature(fn):
        import inspect
        sfn = inspect.signature(fn)
        found_args = False
        found_kwargs = False
        for p in sfn.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                found_args = True
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                found_kwargs = True
        return found_args, found_kwargs

    def _inner(kwargs):
        if isinstance(inputs, (list, tuple)):
            # there is no mapping to be done, so just select the necessary inputs
            input_args = {k: kwargs[k] for k in inputs}
        elif isinstance(inputs, dict):
            input_args = {v: kwargs[k] for k, v in inputs}
        else:
            raise ValueError("'inputs' must be of type list, tuple or dict")

        ret = function(**input_args)

        kwargs = dict(kwargs)
        print(kwargs)

        if isinstance(outputs, (list, tuple)):
            pass
        else:
            ret = [ret]

        for k, v in zip(outputs, ret):
            kwargs[k] = v

        return kwargs


    return _inner
