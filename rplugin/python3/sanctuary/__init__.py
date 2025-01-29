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
        self.ns = nvim.api.create_namespace('sanctuary')


    @pynvim.command('Vsanct', nargs='0', sync=True)
    def create_vertical_buffer(self, args):
        self.nvim.command('vnew')
        buffer = self.nvim.current.buffer
        buffer.name = "Sanctuary"        
        self.nvim.command('setlocal wrap')
        self.nvim.command('setlocal buftype=nofile')
        self.nvim.command('setlocal nonumber')
        self.nvim.command('setlocal noswapfile')
        self.nvim.command('setlocal bufhidden=hide')
        
        self.nvim.command('highlight SanctuaryPrompt guifg=#5555FF')
        
        buffer.append(self.prompt)
        
        self.highlight_prompt(buffer)
        
        self.nvim.command('normal! G$')
        self.nvim.command('inoremap <buffer> <CR> <Esc>:call ProcessSanctuaryInput()<CR>')
        self.nvim.command('nnoremap <buffer> i $a')
        self.nvim.command('nnoremap <buffer> a $a')
        self.nvim.command('startinsert!')

    def highlight_prompt(self, buffer):
        """Highlight the prompt at the current line"""
        last_line = len(buffer) - 1
        self.nvim.api.buf_set_extmark(
            buffer,
            self.ns,
            last_line,
            0,
            {'end_col': len(self.prompt), 'hl_group': 'SanctuaryPrompt'}
        )

    @pynvim.function('ProcessSanctuaryInput')
    def process_input(self, args):
        buffer = self.nvim.current.buffer
        last_line = buffer[-1]
        
        if last_line.startswith(self.prompt):
            user_input = last_line[len(self.prompt):]
            
            response = self.handle_input(user_input)
            
            buffer.append(response)
            buffer.append(self.prompt)
            self.highlight_prompt(buffer)
            
            self.nvim.command('normal! G$')
            self.nvim.command('startinsert!')

    def handle_input(self, input_text : str):
        response = self.agent.ask_question(input_text)        
        lines = response.split('\n')
        return lines

    @pynvim.command('SetNotes', nargs='1', sync=True)
    def set_notes_dir(self, dir):
       return 0 

