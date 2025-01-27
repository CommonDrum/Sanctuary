import pynvim

@pynvim.plugin
class SanctuaryPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim
        # Load configuration from our Lua module when plugin initializes
        self.cfg = nvim.exec_lua('return require("sanctuary").getConfig()')
    
    @pynvim.command('Helloo', nargs='0', sync=True)
    def hello_command(self, args):
        greeting = self.cfg.get('greeting', 'Hello from Sanctuary!')
        self.nvim.command(f"echomsg '{greeting}'")
