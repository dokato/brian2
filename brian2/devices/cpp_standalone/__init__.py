from brian2.core.preferences import brian_prefs
from .codeobject import CPPStandaloneCodeObject
from .device import cpp_standalone_device

brian_prefs['codegen.target'] = CPPStandaloneCodeObject
