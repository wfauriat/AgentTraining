# main.py — CLI entry point
# Owns: user input, printing, conversation history across turns.
# Delegates everything else to loop.run().

from tools import meta, web
import loop

SYSTEM_PROMPT = "You are a helpful assistant running locally on the user's machine."

tools = meta.TOOLS + web.TOOLS
dispatch_table = {
    "get_current_time": meta.dispatch,
    "web_search": web.dispatch,
    "web_fetch": web.dispatch
}

def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Agent ready. Type 'quit' to exit.\n")

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