### Contributing to SMAC3

You are interested in developing a new feature or have found a bug? Awesome, feel welcome and read this guide in order to find out how to best report your ideas so that we can include them as quickly as possible.

### General Notes
We are always happy to read about your experiences and ideas on how to improve SMAC3. We use the issue tracker as our main communication platform, so please open an issue to

* report bugs (label as `bug`)
* ask usage questions (label as `question`)
* discuss feature requests (label as `request`)
* let us know if you plan to contribute code

NOTE: Before working on a new feature please first create an issue. If the relevant issue already exists, please  comment that issue to let us know that you are going to work on it. Through that, we can ensure that there is no duplicated effort.

### New Features

If you are looking for feature that does not yet exist in SMAC3, we are happy to hear about it. Open an issue on our [issues list on GitHub](https://github.com/automl/SMAC3/issues), and describe 
- the feature you would like to see
- why you need it and
- how it should work.

If you already know how to implement the feature, please create a pull request. Please see the [Pull request](#pull-requests) section, to read further details on pull requests.

### <a name="report-bugs"></a> Report Bugs

Open an issue in our [issue list on GitHub](https://github.com/automl/SMAC3/issues).

Before you report a bug, please make sure that:

1. Your bug hasn't already been reported in our [issue tracker](https://github.com/automl/SMAC3/issues).
2. You are using the latest SMAC3 version.

If you found a bug, please provide us the following information:

- A description of the problem
- An example to reproduce the problem
- Any information about your setup that could be helpful to resolve the bug (such as installed python packages)
- Feel free, to add a screenshot showing the issue, if it helps.

### Work on New Features

To work on new features, create a fork of the original repository and implement the feature there. When you are happy with the final result you can create a pull request which we will review.
A good tutorial on how to do this is in the Github Guide: [Fork a repo](https://help.github.com/articles/fork-a-repo/).

### <a name="pull-requests"></a> Pull Requests

The target branch for your pull request has to be the development branch. If you have not worked with pull requests, please take a look at [how to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github) or read more about it in the official github documentation <https://help.github.com/articles/about-pull-requests/>

If you know how to fix a bug or implement your own feature, follow this small guide:

- Check the issue tracker if someone has already reported the same idea or found the same bug.
- Create a pull request, after you have implemented your changes in your fork. Using a separate branch for the fix is required.
- Pull request must include tests.
- We are using the Google Style Python Docstrings. Take a look at this [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). 
- The code should follow the PEP8 coding convention.

We try to react as fast as possible to your pull request.

### Reporting Security Issues

If you find security related issues, vulnerabilities or bugs including sensitive information, please do *not* report this to the issue tracker, or elsewhere public. Instead, please let us know via security@automl.org
