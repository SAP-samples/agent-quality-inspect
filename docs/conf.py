import os
import sys
import inspect

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("../src"))



project = 'agent_inspect'
copyright = '2026, ""'
author = '""'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode",  "sphinx.ext.mathjax", "sphinx.ext.autodoc",     "sphinx_autodoc_typehints",]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    # 'special-members': '__init__',
    'inherited-members': False,
}


# autodoc_typehints = "none"  # turn off typing hit in class
autodoc_typehints = "description"
typehints_use_rich_description = True



def skip_static_methods(app, what, name, obj, skip, options):
    # We'll check classes, methods, functions — anything that can be a staticmethod
    # Only try skipping if 'what' is one of these
    if what not in ("class", "method", "function"):
        return None  # no skip by default for other types

    try:
        # Attempt to get the class object from obj.__qualname__
        qualname = getattr(obj, "__qualname__", "")
        if '.' not in qualname:
            return None

        cls_name = qualname.split('.')[-2]

        # Find the module where this obj is defined
        module = inspect.getmodule(obj)
        if module is None:
            return None

        cls = getattr(module, cls_name, None)
        if cls is None:
            return None

        # Check if this attribute on the class is a staticmethod
        attr = cls.__dict__.get(name)
        if isinstance(attr, staticmethod) and name not in ['get_auc_score_from_progress_scores', 'get_ppt_score_from_progress_scores', 'get_success_score_from_progress_score', 'get_success_score_from_validation_results', 'compute_statistic_analysis_result'] :
            print(attr)
            print(name)
            return True  # skip it!

    except Exception:
        # silently fail and do not skip
        pass

    return None  # default behavior


def skip_methods_by_name(app, what, name, obj, skip, options):
    # We'll check classes, methods, functions — anything that can be skipped
    # Only try skipping if 'what' is one of these
    if what not in ("class", "method", "function"):
        return None  # no skip by default for other types

    try:
        # List of method names to skip (both static and non-static)
        methods_to_skip = [
            'get_system_prompt', 'get_user_message_reflection'
        ]

        # Skip methods based on their name
        if name in methods_to_skip:
            print(f"Skipping method: {name}")
            return True  # Skip this method

    except Exception:
        # Silently fail and do not skip
        pass

    return None  # default behavior




def setup(app):
    app.connect("autodoc-skip-member", skip_static_methods)
    app.connect("autodoc-skip-member", skip_methods_by_name)




templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
