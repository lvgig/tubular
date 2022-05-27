import tubular
from tubular import base
from tubular import capping
from tubular import dates
from tubular import imputers
from tubular import mapping
from tubular import misc
from tubular import nominal
from tubular import numeric
from tubular import strings

from pprint import pprint


class Helper:

    def __init__(self):

        self.module_dict = {
            "base": self._format_module_transformers(base),
            "capping": self._format_module_transformers(capping),
            "dates": self._format_module_transformers(dates),
            "imputers": self._format_module_transformers(imputers),
            "mapping": self._format_module_transformers(mapping),
            "misc": self._format_module_transformers(misc),
            "nominal": self._format_module_transformers(nominal),
            "numeric": self._format_module_transformers(numeric),
            "strings": self._format_module_transformers(strings),
        }

        self.tubular_modules = self._get_tubular_modules()

        self._check_module_coverage()

        self.formatted_dict = self._format_module_dict()

    def _get_tubular_modules(self):
        """Collects the available transformer modules in Tubular"""

        return [
            mod for mod in dir(tubular) if
            not mod.startswith("_") and
            not mod == "helpers" and
            not mod == "list_transformers"
        ]

    def _format_module_transformers(self, module):
        """Formats the dir call of a module to only keep Transformers"""

        return [
            item for item in dir(module) if
            item.endswith("Transformer") or item.endswith("Imputer")
        ]
        
    def _check_module_coverage(self):
        """Check the module_dict attribute covers all available modules"""

        if self.tubular_modules != list(self.module_dict.keys()):
            raise Warning(
                "Not all modules accounted for in tubular.list_transformers(),"
                " please update tubular.helpers.Helper class"
            )

    def _format_module_dict(self):
        """Formats the module_dict to a user friendly cmd output"""

        output_string = ""

        for k, v in self.module_dict.items():

            new_string = f"{k}:"
            for transformer in v:
                new_string += f"\n\t{transformer}"

            new_string += "\n"
            output_string += new_string
        
        return output_string

    def print_format_dict(self):

        print(self.formatted_dict)


def list_transformers():
    """User call for printing available transformers"""

    Helper().print_format_dict()
