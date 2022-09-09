.. D2C documentation master file, created by
   sphinx-quickstart on Thu Sep  8 19:20:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to D2C's documentation!
===============================

**D2C** is a Data-driven Control Library based on reinforcement learning. It is a platform for reinforcement learning experiments and solving various control problems in the real-world Scenario. On the on hand, It makes the RL experiments Fast and convenient. On the other hand, It makes offline reinforcement learning technology be applied in the real world application more possibly and simplistically.

The overall framework of D2C is as below:

.. figure:: ./_static/images/overall_framework.png

The supported RL algorithms include:

* :class:`~d2c.models.TD3BCAgent` `TD3+BC <https://arxiv.org/pdf/2106.06860.pdf>`_
* :class:`~d2c.models.DOGEAgent` `DOGE <https://arxiv.org/abs/2205.11027.pdf>`_

Here are other features of D2C:

* Include a large collection of offline reinforcement learning algorithms: model-free offline RL, model-based offline RL, planning method and imitation learning. There are our self-developed algorithms and other advanced algorithms in each category.
* It is highly modular and extensible. It is easy for you to build custom algorithms and conduct experiments.
* Policy training process is automatic in real-world scenario applications. It simplifies the processes of problem definition, model training, policy evaluation and model deployment.


Installation
------------


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/developer


.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/d2c.data
   api/d2c.envs
   api/d2c.evaluators
   api/d2c.models
   api/d2c.trainers
   api/d2c.utils


Indices and tables
----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
