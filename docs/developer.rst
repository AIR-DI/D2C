Developer Guide
=================

Add RL Algorithm
------------------

The working directory of the RL algorthm module is ``d2c/models``. There are several subfolders to classify the RL algorithms. All the algorithms are inherited from the base class :class:`~d2c.models.base.BaseAgent`.

Know the ``BaseAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^
The system provides the algorithm base class to abstract some essential methods in the algorithm developing, including:

- ``__init__``: there are some essential parameters of the RL algorithms, like ``batch_size``, ``discount``... The detail information of the parameters can be found in doc string of the base class :class:`~d2c.models.base.BaseAgent`. Besides the parameters configuration, It calls two methods to build the networks and some other elements of the algorithm:

    - ``_get_modules``: please see below for details.

    - ``_build_agent``: please see below for details.

- ``_get_modules``: it likes a factory that provides the functions to produce the network models of the algorithm, like actor and Q net. It uses the input ``model_params`` and the network classes in the module :module:`~d2c.utils.networks` to build the network factories. You can add some new networks in :module:`~d2c.utils.networks` as needed.

- ``_build_agent``: it calls some methods of itself to build networks, optimizers, and some Dicts to store the training information and testing policies.

    - Please see below for details of these methods.

- ``_build_fns``: it first uses the class :class:`~d2c.models.base.BaseAgentModule` to build all the network models of the RL algorithm.

    - ``BaseAgentModule``: It is a base class in the module :module:`~d2c.models.base`. In your algorithm module, you should inherit it to build the class ``AgentModule`` of your algorithm. Its initialization parameter ``modules`` is come from the result of the above method :meth:`~d2c.models.base.BaseAgent._get_modules`. Its method :meth:`~d2s.models.base.BaseAgentModule._build_modules` builds the models of the algorithm according to the input network factories. You should implement this method in your ``AgentModule`` class.

    - We get the attribute ``_agent_module`` by instantiating the class ``AgentModule``. Then you can build some attributes as the reference of the network models in ``_agent_module`` for convenience.

- ``_init_vars``: it is not necessary.

- ``_build_optimizers``: the parameters for building the optimizers are in attribute ``_optimizers``. You can use the function :func:`~d2c.utils.utils.get_optimizer` to build all the optimizers needed with the parameters.

- ``_build_loss``:

Implement ``YOUR_ALGORITHM.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configurate your algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

Test your algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit test
.............

Test on benchmark
...............

Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Code style(PEP8)
.................

Annotations(doc string)
................

Type annotations
.................

Develop workflow
.................
