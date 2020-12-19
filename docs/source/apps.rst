****
Apps
****

```{important}
The app classes are currently only manually tested. Unexpected behaviour is
therefore more likely than for classes that are integrated in an automatic
testing framework.
```

.. currentmodule:: erlotinib.apps

Apps are classes that create interactive Dash apps, which allow the exploration
of PKPD models and inference results.

At the moment callbacks have to be implemented by hand, however methods are provided
which simplify this process. Examples can be found in the docstrings of each app.

For further details on Dash apps, please refer to Dash's documentation.

Note that apps have currently only been manually tested.

Base classes:

- :class:`BaseApp`

Overview:

- :class:`PDSimulationController`

.. autoclass:: BaseApp
    :members:

.. autoclass:: PDSimulationController
    :members:
    :inherited-members: