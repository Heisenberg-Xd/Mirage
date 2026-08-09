import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import NavBar from '../components/NavBar';
import ChatPanel from '../components/ChatPanel';
import VerificationPanel from '../components/VerificationPanel';
import { verifyAnswer } from '../api/verification';
import type { ConversationMessage } from '../types';

// ── Types for lifted state coming from App.tsx ─────────────────────────────────

interface WorkspaceProps {
  messages: ConversationMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ConversationMessage[]>>;
  selectedMessageId: string | null;
  setSelectedMessageId: React.Dispatch<React.SetStateAction<string | null>>;
}

// ── Simple ID generator — no external package needed ──────────────────────────

function newId(): string {
  return crypto.randomUUID();
}

// ── Workspace ─────────────────────────────────────────────────────────────────

export default function Workspace({
  messages,
  setMessages,
  selectedMessageId,
  setSelectedMessageId,
}: WorkspaceProps) {
  const location = useLocation();

  // Guard against React 18 StrictMode double-invoking the effect.
  // Without this, initialQuestion fires twice → duplicate messages.
  const initialFired = useRef(false);

  /**
   * If the user came from the landing page hero search, navigate passes
   * { state: { initialQuestion } }. Fire it exactly once on mount.
   */
  useEffect(() => {
    if (initialFired.current) return;
    const state = location.state as { initialQuestion?: string } | null;
    if (state?.initialQuestion) {
      initialFired.current = true;
      handleSendMessage(state.initialQuestion);
      // Clear router state so back-button / re-renders don't re-fire.
      window.history.replaceState({}, '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Core send / verify logic ────────────────────────────────────────────────

  const handleSendMessage = async (text: string) => {
    const newMessageId = newId();
    const newMessage: ConversationMessage = {
      id: newMessageId,
      question: text,
      raw_answer: null,
      result: null,
      status: 'loading',
      timestamp: new Date(),
    };

    // Append the new message and immediately select it for the right panel.
    setMessages((prev) => [...prev, newMessage]);
    setSelectedMessageId(newMessageId);

    try {
      const data = await verifyAnswer(text);
      // Update only this specific message by ID — never replace the array.
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === newMessageId
            ? { ...msg, raw_answer: data.raw_answer, result: data.result, status: 'complete' }
            : msg
        )
      );
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === newMessageId
            ? {
                ...msg,
                status: 'error',
                errorMessage:
                  error instanceof Error ? error.message : 'An unknown error occurred',
              }
            : msg
        )
      );
    }
  };

  // ── Derived values ──────────────────────────────────────────────────────────

  const isVerifying = messages.some((m) => m.status === 'loading');
  const selectedMessage = messages.find((m) => m.id === selectedMessageId) ?? null;

  // ── Render ──────────────────────────────────────────────────────────────────
  //
  // Layout (100vh flex column):
  //   ┌─────────── NavBar (fixed 64px) ───────────┐
  //   │   ChatPanel (flex:1, overflow)  │ VPanel  │
  //   └───────────────────────────────────────────┘
  //
  // The outer wrapper is h-screen flex-col.
  // The content row below the navbar is flex-1 overflow-hidden.
  // ChatPanel internally is also flex-col so the scroll area can grow freely.

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      {/*
        NavBar uses position:fixed — it is removed from document flow, so
        without this spacer the content row would start at top:0 and be
        covered by the navbar.  Height must match --navbar-height (64px).
      */}
      <div className="shrink-0" style={{ height: 'var(--navbar-height)' }} aria-hidden />

      {/* Content area — fills everything below the fixed navbar */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left — Chat column (65%) */}
        <div className="flex-1 min-w-0 border-r-2 border-[#111111] overflow-hidden">
          <ChatPanel
            messages={messages}
            selectedMessageId={selectedMessageId}
            onSelectMessage={setSelectedMessageId}
            onSendMessage={handleSendMessage}
            isLoading={isVerifying}
          />
        </div>

        {/* Right — Verification panel (35%), desktop only */}
        <div
          className="hidden lg:flex flex-col bg-[#FAF8F4] overflow-hidden"
          style={{ width: '35%', minWidth: '280px', maxWidth: '480px' }}
        >
          <VerificationPanel message={selectedMessage} />
        </div>
      </div>
    </div>
  );
}
