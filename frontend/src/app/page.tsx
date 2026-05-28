import { ChatInterface } from "@/components/chat/ChatInterface";

export default function ChatPage() {
  return (
    <section aria-label="Chat" className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
        <p className="text-sm text-fg-muted">
          Query your indexed documents. Retrieval and the LLM run on your local backend.
        </p>
      </header>
      <ChatInterface />
    </section>
  );
}
