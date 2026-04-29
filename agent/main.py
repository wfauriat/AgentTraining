# main.py — CLI entry point
# Owns: user input, printing, conversation history across turns.
# Delegates everything else to loop.run().

from tools import meta, web, files, python_sandbox, docs
import loop

SYSTEM_PROMPT = (
    "You are a helpful assistant running locally on the user's machine. "
    "You have access to tools. When you have completed a user's request, "
    "call the 'finish' tool with a summary message rather than replying "
    "with regular text."
)


def main():
    # Tool wiring: schema list goes to the model, dispatch table to the loop.
    # When you add a new tool module, register it in both places.
    tools = meta.TOOLS + web.TOOLS + files.TOOLS + python_sandbox.TOOLS + docs.TOOLS
    dispatch_table = {
        "get_current_time": meta.dispatch,
        "web_search":       web.dispatch,
        "web_fetch":        web.dispatch,
        "read_file":        files.dispatch,
        "write_file":       files.dispatch,
        "run_python":       python_sandbox.dispatch,
        "search_documents": docs.dispatch,
        "finish":           meta.dispatch
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Agent ready. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "quit":
            print("Powering down.")
            break

        if not query:
            continue

        messages.append({"role": "user", "content": query})

        reply = loop.run(messages, tools, dispatch_table)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()