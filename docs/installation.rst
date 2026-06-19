.. highlight:: shell

============
Installation
============


Stable release
--------------

To install soap, run this command in your terminal:

.. code-block:: console

    $ uv pip install soapcw

This is the preferred method to install soap, as it will always install the most recent stable release.

If you don't have `uv`_ installed, the `uv installation guide`_ can guide
you through the process. (You can also use plain ``pip install soapcw`` if you prefer.)

.. _uv: https://docs.astral.sh/uv/
.. _uv installation guide: https://docs.astral.sh/uv/getting-started/installation/


From sources
------------

The sources for soap can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/jcbayley/soapcw

Once you have a copy of the source, the easiest way to set up an environment and
install the project (along with all its dependencies, pinned from ``uv.lock``) is:

.. code-block:: console

    $ cd soapcw
    $ uv sync

This creates a ``.venv`` and installs soap in editable mode. Run commands inside
the environment with ``uv run``, for example:

.. code-block:: console

    $ uv run soapcw-run-soap-astro --help

Alternatively, you can install into an existing environment with:

.. code-block:: console

    $ uv pip install .


.. _Github repo: https://github.com/jcbayley/soapcw
