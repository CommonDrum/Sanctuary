-- C:/Users/Karol/Dev/Sanctuary/lua/sanctuary.lua

local M = {}

function M.setup()
    -- This function gets called when your plugin loads
    -- We can add any Lua-side initialization here
    
    -- Register our commands in a way that works with Neovim
    vim.api.nvim_create_user_command('StartChatVertical', function()
        -- This ensures the Python command is available
        vim.cmd('StartChatVertical')
    end, {})

    vim.api.nvim_create_user_command('StartChatHorizontal', function()
        vim.cmd('StartChatHorizontal')
    end, {})

    -- Let users know the plugin is ready
    vim.notify("Sanctuary chat plugin initialized!", vim.log.levels.INFO)
end

return M