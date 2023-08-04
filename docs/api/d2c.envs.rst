d2c.envs
================


BaseEnv
-------------

.. autoclass:: d2c.envs.BaseEnv
   :members:
   :undoc-members:
   :show-inheritance:


.. _external-env-label:

External Env
---------------

D4rlEnv
~~~~~~~~~~~~~~

.. autoclass:: d2c.envs.external.D4rlEnv
   :members:
   :undoc-members:
   :show-inheritance:

.. _benchmark-env-label:

benchmark_env
~~~~~~~~~~~~~~

.. autofunction:: d2c.envs.benchmark_env


.. _learned-env-label:

Learned Env
---------------

LeaEnv
~~~~~~~~~~~~~~

.. autoclass:: d2c.envs.LeaEnv
   :members:
   :undoc-members:
   :show-inheritance:

Dynamics
~~~~~~~~~~~~~~

BaseDyna
^^^^^^^^^^^^^^^

.. autoclass:: d2c.envs.learned.dynamics.BaseDyna
   :members:
   :undoc-members:
   :show-inheritance:

ProbDyna
^^^^^^^^^^^^^^^

.. autoclass:: d2c.envs.learned.dynamics.ProbDyna
   :members:
   :undoc-members:
   :show-inheritance:

.. _register-dyna-label:

register_dyna
^^^^^^^^^^^^^^^

.. autofunction:: d2c.envs.learned.dynamics.register_dyna

.. _make-dynamics-label:

make_dynamics
^^^^^^^^^^^^^^^

.. autofunction:: d2c.envs.learned.dynamics.make_dynamics