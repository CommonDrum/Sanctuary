import subprocess
import time
import ollama
import pynvim
from agent import Agent

@pynvim.plugin
class ChatPlugin(object):
    def __init__(self, nvim):

        model_name = "llama3.2:3b"

        try:
            ollama.pull(model_name)
        except Exception as e:
            print(f"Error: {e}")
    self.nvim = nvim
    self.chat_buffer = None
    self.input_buffer = None
    self.agent = Agent()

def create_split(self, split_type='vsplit'):
    self.nvim.command(split_type)
    self.chat_buffer = self.nvim.create_buf(True, True)
    self.chat_buffer.name = 'Chat'
    self.nvim.current.window.buffer = self.chat_buffer

@pynvim.command('StartChatVertical')
def start_chat_vertical(self):
    self.create_split('vsplit')
    self.setup_input()

@pynvim.command('StartChatHorizontal') 
def start_chat_horizontal(self):
    self.create_split('split')
    self.setup_input()

def setup_input(self):
    # Input window setup code here
    pass
