.. D2C documentation master file, created by
   sphinx-quickstart on Thu Sep  8 19:20:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to D2C's documentation!
===============================

D2C(**D**\ ata-\ **d**\ riven **C**\ ontrol Library) is a library for data-driven control based on reinforcement learning. It is a platform for conducting reinforcement learning experiments and solving various control problems in real-world scenarios. It has two main advantages: first, it makes the RL experiments fast and convenient; second, it enables the application of offline reinforcement learning technology in real-world settings more easily and simply.

The overall framework of D2C is as below:

.. figure:: ./_static/images/overall_framework.png

The supported RL algorithms include:

* :class:`~d2c.models.TD3BCAgent` `TD3+BC <https://arxiv.org/pdf/2106.06860.pdf>`_

* :class:`~d2c.models.DOGEAgent` `DOGE <https://arxiv.org/abs/2205.11027.pdf>`_

* :class:`~d2c.models.H2OAgent` `H2O <https://arxiv.org/abs/2206.13464.pdf>`_

* :class:`~d2c.models.IQLAgent` `IQL <https://arxiv.org/pdf/2110.06169.pdf>`_

* :class:`~d2c.models.DMILAgent` `DMIL <https://arxiv.org/abs/2207.00244>`_

* :class:`~d2c.models.BCAgent` `BC <http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf>`_

Here are other features of D2C:

* It includes a large collection of offline reinforcement learning algorithms: model-free offline RL, model-based offline RL, planning methods and imitation learning. In each category, there are our self-developed algorithms and other advanced algorithms.

* It is highly modular and extensible. You can easily build custom algorithms and conduct experiments with it.

* It automates the policy training process in real-world scenario applications. It simplifies the steps of problem definition, model training, policy evaluation and model deployment.


Installation
------------
D2C interface can be installed as follows:

.. code-block:: bash

    $ git clone https://github.com/AIR-DI/D2C.git
    $ cd d2c
    $ pip install -e .


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/overview
   tutorials/create_dataset
   tutorials/customize_environment
   tutorials/developer
   tutorials/configuration


.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/d2c.data
   api/d2c.envs
   api/d2c.evaluators
   api/d2c.models
   api/d2c.trainers
   api/d2c.utils


.. toctree::
   :maxdepth: 1
   :caption: Community

   contributor


Indices and tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
