#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#
# This file has been adapted from PINTS (https://github.com/pints-team/pints/)
# which is released under the BSD 3-clause license.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import argparse
import datetime
import os
import re
import subprocess
import sys
import unittest


def run_copyright_checks():
    """
    Checks that the copyright year in LICENSE.md is up-to-date and that each
    file contains the copyright header
    """
    print('\nChecking that copyright is up-to-date and complete.')

    year_check = True
    current_year = str(datetime.datetime.now().year)

    with open('LICENSE.md', 'r') as license_file:
        license_text = license_file.read()
        if 'Copyright (c) ' + current_year in license_text:
            print("Copyright notice in LICENSE.md is up-to-date.")
        else:
            print('Copyright notice in LICENSE.md is NOT up-to-date.')
            year_check = False

    # Recursively walk the pkpd directory and check copyright header is in
    # each checked file type
    header_check = True
    checked_file_types = ['.py']
    copyright_header = """#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#"""

    for dirname, _, file_list in os.walk('chi'):
        for f_name in file_list:
            if any([f_name.endswith(x) for x in checked_file_types]):
                path = os.path.join(dirname, f_name)
                with open(path, 'r') as f:
                    if copyright_header not in f.read():
                        print('Copyright blurb missing from ' + path)
                        header_check = False

    if header_check:
        print('All files contain copyright header.')

    if not year_check or not header_check:
        print('FAILED')
        sys.exit(1)


def run_doctests():
    """
    Runs a number of tests related to documentation.
    """

    print('\n{}\n# Starting doctests... #\n{}\n'.format('#' * 24, '#' * 24))

    # Check that documentation can be built with sphinx
    doctest_sphinx()

    # Check that all classes and methods are documented in rst files, and no
    # unintended modules are exposed via a public interface
    doctest_rst_and_public_interface()

    # Check that example code runs without throwing errors
    doctest_example_code()

    print('\n{}\n# Doctests passed. #\n{}\n'.format('#' * 20, '#' * 20))


def doctest_sphinx():
    """
    Runs sphinx-build in a subprocess, checking that it can be invoked without
    producing errors.
    """
    print('Checking if docs can be built.')
    p = subprocess.Popen([
        'sphinx-build',
        '-b',
        'doctest',
        'docs/source',
        'docs/build/html',
        '-W',
    ])
    try:
        ret = p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()
        print('')
        sys.exit(1)
    if ret != 0:
        print('FAILED')
        sys.exit(ret)


def doctest_rst_and_public_interface():
    """
    Check that every class and method is documented in an rst file and that
    no unintended modules are exposed via a public interface.
    """
    print('\nChecking that all classes and methods are documented in an RST '
          'file and that public interfaces are clean.')

    import chi
    import chi.library
    import chi.plots

    # If any modules other than these are exposed it may indicate that a module
    # has been inadvertently exposed in a public context, or that a new module
    # has been added to pints and should be imported above and included in this
    # list.
    chi_submodules = ['chi.library', 'chi.plots']

    doc_symbols = get_all_documented_symbols()

    check_exposed_symbols(chi, chi_submodules, doc_symbols)
    check_exposed_symbols(chi.plots, [], doc_symbols)

    print('All classes and methods are documented in an RST file, and all '
          'public interfaces are clean.')


def doctest_example_code():
    """
    Check that example scripts are compatible with chi interface and run
    without errors.
    """
    print('\nChecking that all example scripts are compatible with chi.')

    script_dir = \
        os.path.dirname(os.path.abspath(__file__)) \
        + '/docs/source/getting_started/code'
    scripts = os.listdir(script_dir)
    for script in scripts:
        script = script_dir + '/' + script
        p = subprocess.Popen([
            'python',
            script,
            '--test',
        ])
        try:
            ret = p.wait()
        except KeyboardInterrupt:
            try:
                p.terminate()
            except OSError:
                pass
            p.wait()
            print('')
            sys.exit(1)
        if ret != 0:
            print('FAILED')
            sys.exit(ret)

    print('All example scripts are compatible with chi.')


def get_all_documented_symbols():
    """
    Recursively traverse docs/source and identify all autoclass and
    autofunction declarations.
    Returns: A dict containing a list of classes and a list of functions
    """

    doc_files = []
    for root, dirs, files in os.walk(os.path.join('docs', 'source')):
        for file in files:
            if file.endswith('.rst'):
                doc_files.append(os.path.join(root, file))

    # Regular expression that would find either 'module' or 'currentmodule':
    # this needs to be prepended to the symbols as x.y.z != x.z
    regex_module = re.compile(r'\.\.\s*\S*module\:\:\s*(\S+)')

    # Regular expressions to find autoclass and autofunction specifiers
    regex_class = re.compile(r'\.\.\s*autoclass\:\:\s*(\S+)')
    regex_funct = re.compile(r'\.\.\s*autofunction\:\:\s*(\S+)')

    # Identify all instances of autoclass and autofunction in all rst files
    documented_symbols = {'classes': [], 'functions': []}
    for doc_file in doc_files:
        with open(doc_file, 'r') as f:
            # We need to identify which module each class or function is in
            module = ''
            for line in f.readlines():
                m_match = re.search(regex_module, line)
                c_match = re.search(regex_class, line)
                f_match = re.search(regex_funct, line)
                if m_match:
                    module = m_match.group(1) + '.'
                elif c_match:
                    documented_symbols['classes'].append(
                        module + c_match.group(1))
                elif f_match:
                    documented_symbols['functions'].append(
                        module + f_match.group(1))

    # Validate the list for any duplicate documentation
    for symbols in documented_symbols.values():
        if len(set(symbols)) != len(symbols):
            print('The following symbols are unexpectedly documented multiple '
                  'times in rst files:')

            dupes = set([d for d in symbols if symbols.count(d) > 1])
            for d in dupes:
                print('  multiple entries in docs: ' + d)

            print('FAILED')
            sys.exit(1)

    return documented_symbols


def check_exposed_symbols(module, submodule_names, doc_symbols):
    """
    Check ``module`` for any classes and methods not contained in
    ``doc_symbols``, and check for any modules not contained in
    ``submodule_names``.
    Arguments:
    ``module``
        The module to check
    ``submodule_names``
        List of submodules expected to be exposed by ``module``
    ``doc_symbols``
        Dictionary containing lists of documented classes and functions
    """

    import inspect
    exposed_symbols = [x for x in dir(module) if not x.startswith('_')]
    symbols = [getattr(module, x) for x in exposed_symbols]

    classes = [x for x in symbols if inspect.isclass(x)]
    functions = [x for x in symbols if inspect.isfunction(x)]

    # Check for modules: these should match perfectly with _submodule_names
    exposed_modules = [x for x in symbols if inspect.ismodule(x)]
    unexpected_modules = [m for m in exposed_modules if
                          m.__name__ not in submodule_names]

    if len(unexpected_modules) > 0:
        print('The following modules are unexpectedly exposed in the public '
              'interface of %s:' % module.__name__)
        for m in sorted(unexpected_modules):
            print('  unexpected module: ' + m.__name__)

        print('For python modules such as numpy you may need to confine the '
              'import to the function scope. If you have created a new pints '
              'submodule, you will need to make %s (doctest) aware of this.'
              % __file__)
        print('FAILED')
        sys.exit(1)

    # Check that all classes are documented
    undocumented_classes = []
    for _class in classes:
        class_name = module.__name__ + '.' + _class.__name__
        if class_name not in doc_symbols['classes']:
            undocumented_classes.append(class_name)

    if len(undocumented_classes) > 0:
        print('The following classes do not appear in any RST file:')
        for m in sorted(undocumented_classes):
            print('  undocumented class: ' + m)
        print('FAILED')
        sys.exit(1)

    # Check that all functions are documented
    undocumented_functions = []
    for _funct in functions:
        funct_name = module.__name__ + '.' + _funct.__name__
        if funct_name not in doc_symbols['functions']:
            undocumented_functions.append(funct_name)

    if len(undocumented_functions) > 0:
        print('The following functions do not appear in any RST file:')
        for m in sorted(undocumented_functions):
            print('  undocumented function: ' + m)
        print('FAILED')
        sys.exit(1)


def run_unit_tests():
    """
    Runs unit tests.
    """
    tests = os.path.join('chi', 'tests')
    suite = unittest.defaultTestLoader.discover(tests, pattern='test_*.py')
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run tests for chi.',
        epilog='To run individual unit tests, use e.g.'
               ' $ chi/tests/test_likelihoods.py',
    )
    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using the `python` interpreter.',)

    # Copyright checks
    parser.add_argument(
        '--copyright',
        action='store_true',
        help='Check copyright runs to the current year',)

    # Doctests
    parser.add_argument(
        '--doctest',
        action='store_true',
        help='Run any doctests, check if docs can be built',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False

    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()

    # Copyright checks
    if args.copyright:
        has_run = True
        run_copyright_checks()

    # Doctests
    if args.doctest:
        has_run = True
        run_doctests()

    # Help
    if not has_run:
        parser.print_help()
