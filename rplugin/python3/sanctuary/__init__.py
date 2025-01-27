import pynvim

@pynvim.plugin
class SanctuaryPlugin(object):
    def __init__(self, nvim):
        self.nvim = nvim
        self.cfg = nvim.exec_lua('return require("sanctuary").getConfig()')
        self.prompt = ">>> "
        self.input_line = ""

    @pynvim.command('Vbuffer', nargs='0', sync=True)
    def create_vertical_buffer(self, args):
        # Create vertical split
        self.nvim.command('vnew')
        buffer = self.nvim.current.buffer
        
        # Set buffer options
        self.nvim.command('setlocal buftype=nofile')
        self.nvim.command('setlocal nonumber')
        self.nvim.command('setlocal noswapfile')
        self.nvim.command('setlocal bufhidden=hide')
        
        # Initialize buffer with prompt
        buffer.append(self.prompt)
        
        # Move cursor to the end
        self.nvim.command('normal! G$')
        
        # Set up mappings for this buffer
        self.nvim.command('inoremap <buffer> <CR> <Esc>:call ProcessSanctuaryInput()<CR>')
        self.nvim.command('nnoremap <buffer> i $a')
        self.nvim.command('nnoremap <buffer> a $a')
        
        # Enter insert mode
        self.nvim.command('startinsert!')

    @pynvim.function('ProcessSanctuaryInput')
    def process_input(self, args):
        buffer = self.nvim.current.buffer
        # Get the last line (user input)
        last_line = buffer[-1]
        
        if last_line.startswith(self.prompt):
            # Extract user input (remove prompt)
            user_input = last_line[len(self.prompt):]
            
            # Process the input and get response
            response = self.handle_input(user_input)
            
            # Add response and new prompt
            buffer.append(response)
            buffer.append(self.prompt)
            
            # Move cursor to the end
            self.nvim.command('normal! G$')
            self.nvim.command('startinsert!')

    def handle_input(self, input_text):
        # Basic command handling - you can expand this
        if input_text.lower() == 'hello':
            return "Hello there!"
        elif input_text.lower() == 'help':
            return "Available commands: hello, help, quit"
        elif input_text.lower() == 'quit':
            self.nvim.command('quit')
            return ""
        else:
            return f"You said: {input_text}"
