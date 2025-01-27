local config = {}

local function setup(cfg)
    config = cfg
    -- Add a basic keymap to trigger our sanctuary command
    vim.keymap.set('n', '<Leader>s', ':call SanctuaryHello()<CR>', { silent = true })
end

-- This function will be called from Python to get config
local function getConfig()
    return config
end

return { setup = setup, getConfig = getConfig }
