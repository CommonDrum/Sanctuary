import pynvim

@pynvim.plugin
class SanctuaryPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim
        # Get config from Lua
        self.cfg = nvim.exec_lua('return require("sanctuary").getConfig()')

    @pynvim.function("SanctuaryHello", sync=True)
    def hello(self, args):
        self.nvim.command("echomsg 'Hello from Sanctuary!'")ello from Sanctuary plugin!\n")m Sanctuary plugin!\n")  pass
