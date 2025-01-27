-- Store configuration in a local table with default values
local config = {
    greeting = "Default greeting from Sanctuary!"
}

-- Setup function that users will call from their init.lua/init.vim
local function setup(user_config)
    config = vim.tbl_extend("force", config, user_config or {})
end

-- This function allows our Python code to access the configuration
local function getConfig()
    return config
end

-- Export our functions
return {
    setup = setup,
    getConfig = getConfig,
    echo_buffer = echo_buffer
}
