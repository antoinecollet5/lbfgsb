.. highlight:: shell

============
Contributing
============

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Reporting Bugs and Defects
~~~~~~~~~~~~~~~~~~~~~~~~~

A defect is any variance between the actual and expected behavior. This includes bugs in the code, issues in the documentation, or problems with visualizations.

Please report defects using the `GitHub Issue Tracker <https://github.com/antoinecollet5/lbfgsb/issues>`_.
When possible, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Relevant logs, error messages, or screenshots
- Information about your environment (OS, Python version, lbfgsb version)

See the `Pull Request Guidelines`_ section for details on development best practices.

Features
~~~~~~~~

If you would like to propose a new feature, please open an issue on the
`GitHub Issue Tracker <https://github.com/antoinecollet5/lbfgsb/issues>`_.

Community members and maintainers will help refine and discuss your idea before implementation.
Early discussion helps ensure that proposed features align with the project’s goals and scope.

Please see the `Pull Request Guidelines`_ section for implementation details.

Documentation
~~~~~~~~~~~~~

The project can always use more documentation, whether as part of the official
lbfgsb documentation, in docstrings, tutorials, or external resources such as blog posts and articles.

For docstrings, please use the
`NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Working on Issues
-----------------

Once an issue is created, its progress is tracked on GitHub using
`labels <https://github.com/antoinecollet5/lbfgsb/labels>`_ and milestones.

When an issue is ready for review, a **Pull Request** can be opened and linked
to the corresponding issue.

Pull Request Guidelines
-----------------------

Please submit pull requests against the **develop** branch (not the **master** branch).
Each pull request should be self-contained and address a single issue or feature.

Before submitting a pull request, please ensure that:

1. **Tests**
   - New code is fully tested.
   - Run the test suite with coverage to identify gaps.

2. **Documentation**
   - Relevant documentation is updated.
   - Docstrings follow the NumPy docstring format.
   - Tutorials and user documentation are updated when necessary.

3. **Continuous Integration**
   - All CI checks pass.
   - All pre-commit hoocks succeed.
   - Code coverage does not decrease.
   - Documentation builds successfully.

4. **Clarity**
   - The pull request clearly describes the changes made.
   - The description explains how the changes address the issue.
   - Screenshots or examples are encouraged when they improve clarity.

Once submitted, maintainers will review your pull request and may request changes.
After approval, a maintainer will merge it.

Thank you very much for your contribution and for helping improve lbfgsb!

Setting Up lbfgsb for Local Development
---------------------------------------

Ready to contribute? Here’s how to set up `lbfgsb` for local development.

1. Fork the `lbfgsb` repository on GitHub.

2. Clone your fork locally::

    $ git clone git@github.com:your_username/lbfgsb.git
    $ cd lbfgsb

3. Create and activate a virtual environment, then install dependencies::

    $ python -m venv venv
    $ source venv/bin/activate  # On Windows: venv\Scripts\activate
    $ pip install -e .[all]

4. Create a new branch for your work::

    $ git checkout -b name-of-your-bugfix-or-feature

5. Make your changes locally.

6. Run linting, tests, and coverage checks against all supported python versions::

    $ pre-commit run --all-files
    $ make coverage
    $ tox

   If you worked on documentation, you can build it locally:

   .. code-block:: shell

      $ cd docs
      $ make html

   The built documentation will be located in ``docs/build/html``.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Clear and descriptive commit message"
    $ git push origin name-of-your-bugfix-or-feature
