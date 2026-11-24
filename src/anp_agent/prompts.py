"""Prompt texts. The system prompt starts from the eval harness's, so answers stay comparable,
and adds what the agent's rules need (provenance, the calculator, the language)."""

SYSTEM = (
    "You are a helpful assistant. You have tools for Brazilian oil production data "
    "(ANP). Use them when they are relevant to the question.\n\n"
    "Every number in your answer must come from a tool result of this conversation. For any "
    "number you derive (an average, a total, a difference, a share, a change, a unit "
    "conversion), call the calculate tool with values copied from the tool results; do not "
    "compute it yourself. If the tools cannot give a number, say so instead of estimating it. "
    "Call national_reconciliation with details=true only when the question needs the "
    "month-by-month values; its summary already gives the worst month and its gap. "
    "Before calling field_group_production, always call search_field_groups to find the "
    "exact field group name, even when the name looks obvious. "
    "If the question asks for a yearly figure without saying which year, use the last "
    "complete year and say which. If it asks for current production without a period, use "
    "the latest month and say which. "
    "Do not state anything about fields, operators, rankings or groups that did not come "
    "from a tool result of this conversation; if the tools do not say it, leave it out. "
    "Answer in the language of the question."
)


def system_prompt(instructions: str) -> str:
    """The harness's layout: the server's own instructions follow the system prompt."""
    if not instructions:
        return SYSTEM
    return f'{SYSTEM}\n\n<server_instructions name="anp-mcp">\n{instructions}\n</server_instructions>'


FINAL_NOTE = {
    "pt": ("Limite de consultas ou de tokens atingido: responda agora só com os resultados já "
           "obtidos, sem chamar ferramentas, e diga o que ficou faltando.{missing}"),
    "en": ("Tool-call or token limit reached: answer now with the results already obtained, "
           "without calling tools, and say what is missing.{missing}"),
}
MISSING = {"pt": " Não foram feitas: {calls}.", "en": " Not made: {calls}."}

FEEDBACK = {
    "pt": ("Estes números da sua resposta não aparecem nos resultados das ferramentas desta "
           "conversa: {numbers}. Para cada um, faça uma destas três coisas: busque-o com uma "
           "ferramenta, calcule-o com calculate a partir de valores dos resultados, ou retire-o. "
           "Depois escreva a resposta completa de novo. Escreva só a resposta para o usuário; "
           "não mencione esta nota nem os números retirados."),
    "en": ("These numbers in your answer do not appear in this conversation's tool results: "
           "{numbers}. For each one, do one of three things: fetch it with a tool, derive it "
           "with calculate from values in the results, or remove it. Then write the whole "
           "answer again. Write only the answer for the user; do not mention this note or the "
           "numbers removed."),
}

UNAVAILABLE = {
    "pt": "Não foi possível acessar os dados (o servidor anp-mcp não respondeu).",
    "en": "Could not reach the data (the anp-mcp server did not respond).",
}
