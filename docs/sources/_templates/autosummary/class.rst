..
   Template for the html class rendering

   Modified from
   https://github.com/sphinx-doc/sphinx/tree/master/sphinx/ext/autosummary/templates/autosummary/class.rst

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

   {% if attributes %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
      :nosignatures:
      :toctree: generated

      {% for item in attributes if not item.startswith('_') %}
      ~{{ objname }}.{{ item }}
      {% endfor %}
   {% endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      :toctree: generated

      {% for item in methods if not item.startswith('_') %}
      ~{{ objname }}.{{ item }}
      {% endfor %}
   {% endif %}

   {% if examples %}
   .. rubric:: {{ _('Examples') }}

   .. code-block:: python

      {% for line in examples %}
      {{ line }}
      {% endfor %}
   {% endif %}
