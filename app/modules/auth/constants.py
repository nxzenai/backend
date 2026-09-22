SUPER_ADMIN = "super_admin"
ADMIN = "admin"
USER = "user"
TRAINER = "trainer"
TRAINEE = "trainee"
GUEST = "guest"

MODULES = {
    "dashboard", "python_lab", "sql_lab", "eda", "automl", "autodl",
    "autonlp", "genai", "agentic_ai", "crm", "leads",
    "user_management", "platform",
}

ROLE_DEFAULT_MODULES = {
    SUPER_ADMIN: MODULES,
    ADMIN: MODULES - {"platform"},
    TRAINER: {"dashboard"},
    TRAINEE: {"dashboard"},
    GUEST: {"dashboard", "python_lab", "sql_lab", "eda", "automl", "autodl", "autonlp"},
    # Backward compatibility for accounts created before access-control rollout.
    USER: {"dashboard", "python_lab", "sql_lab", "eda", "automl", "autodl", "autonlp", "genai", "agentic_ai"},
}

SUPER_ADMIN_EMAILS = {
    "bhargav@nxzenai.com",
    "fayaz@nxzenai.com",
    "roushan@nxzenai.com",
    "shruthi.n@nxzenai.com",
}
