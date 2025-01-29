local config = {
    greeting = "Default greeting from Sanctuary!"
}

local function setup(user_config)
    config = vim.tbl_extend("force", config, user_config or {})
end

local function getConfig()
    return config
end

return {
    setup = setup,
    getConfig = getConfig,
    echo_buffer = echo_buffer
}
