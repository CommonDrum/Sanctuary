import pynvim
from .agent import Agent

@pynvim.plugin
class SanctuaryPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim
        self.cfg = nvim.exec_lua('return require("sanctuary").getConfig()')
        self.prompt = ">>> "
        self.input_line = ""
        self.agent = Agent()
        self.counter = 0

    @pynvim.command('Vsanct', nargs='0', sync=True)
    def create_vertical_buffer(self, args):
        self.nvim.command('vnew')
        buffer = self.nvim.current.buffer
        
        self.nvim.command('setlocal wrap')
        self.nvim.command('setlocal buftype=nofile')
        self.nvim.command('setlocal nonumber')
        self.nvim.command('setlocal noswapfile')
        self.nvim.command('setlocal bufhidden=hide')
        
        buffer.append(self.prompt)
        
        self.nvim.command('normal! G$')
        
        self.nvim.command('inoremap <buffer> <CR> <Esc>:call ProcessSanctuaryInput()<CR>')
        self.nvim.command('nnoremap <buffer> i $a')
        self.nvim.command('nnoremap <buffer> a $a')
        
        self.nvim.command('startinsert!')

    @pynvim.function('ProcessSanctuaryInput')
    def process_input(self, args):
        buffer = self.nvim.current.buffer
        last_line = buffer[-1]
        
        if last_line.startswith(self.prompt):
            user_input = last_line[len(self.prompt):]
            
            response = self.handle_input(user_input)
            
            buffer.append(response)
            buffer.append(self.prompt)
            
            self.nvim.command('normal! G$')
            self.nvim.command('startinsert!')

    def handle_input(self, input_text : str):
        response = self.agent.ask_question(input_text)        
        lines = response.split('\n')
        return lines
