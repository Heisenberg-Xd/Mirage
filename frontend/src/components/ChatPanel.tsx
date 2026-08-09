import { useRef, useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import type { ConversationMessage } from '../types';

interface ChatPanelProps {
  messages: ConversationMessage[];
  selectedMessageId: string | null;
  onSelectMessage: (id: string) => void;
  onSendMessage: (text: string) => void;
  isLoading: boolean;
}

export default function ChatPanel({
  messages,
  selectedMessageId,
  onSelectMessage,
  onSendMessage,
  isLoading,
}: ChatPanelProps) {
  const scrollRef      = useRef<HTMLDivElement>(null);   // scroll container
  const messagesEndRef = useRef<HTMLDivElement>(null);   // sentinel at bottom
  const inputRef       = useRef<HTMLTextAreaElement>(null);
  const [inputVal, setInputVal]   = useState('');
  const [copiedId, setCopiedId]   = useState<string | null>(null);
  // Track the message count so we know when the user just sent a new one
  const prevMsgCountRef = useRef(0);

  // ── Smart auto-scroll ──────────────────────────────────────────────────────
  // Only scroll to bottom when:
  //   a) A new user message was just appended (user sent a question), OR
  //   b) A response arrived AND the user is already near the bottom (≤120px away).
  // This prevents hijacking the viewport when the user is reading older messages.
  const isNearBottom = () => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= 120;
  };

  const scrollToBottom = (smooth = true) => {
    messagesEndRef.current?.scrollIntoView({
      behavior: smooth ? 'smooth' : 'instant',
      block: 'end',
    });
  };

  useEffect(() => {
    const newCount = messages.length;
    const prevCount = prevMsgCountRef.current;
    prevMsgCountRef.current = newCount;

    if (newCount === 0) return;

    // New message just appended (user sent a question) — always scroll.
    // Response arrived — only scroll if the user is near the bottom.
    const userJustSent = newCount > prevCount;
    if (userJustSent || isNearBottom()) {
      requestAnimationFrame(() => scrollToBottom());
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, isLoading]);

  // ── Re-focus input when loading finishes so user can keep typing ───────────
  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);

  // ── Auto-resize textarea ───────────────────────────────────────────────────
  const resizeTextarea = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 192) + 'px';
  }, []);

  // ── Send handler ───────────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    const text = inputVal.trim();
    if (!text || isLoading) return;
    onSendMessage(text);
    setInputVal('');
    // Reset textarea height after clearing
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
  }, [inputVal, isLoading, onSendMessage]);

  // Enter = send, Shift+Enter = new line
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Copy handler ───────────────────────────────────────────────────────────
  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // Clipboard API unavailable — silent fail
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  //
  // Flex column that fills its parent (parent is flex-1 from Workspace).
  //   ┌─────── Scrollable messages (flex:1, min-height:0) ───────┐
  //   │                                                           │
  //   └─────── Sticky composer (position:sticky, bottom:0) ──────┘

  return (
    <div className="flex flex-col h-full bg-white">

      {/* ── Scrollable messages area ──────────────────────────────────── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar px-4 sm:px-8 pt-8 pb-6"
        style={{ minHeight: 0 }}
      >
        <div className="max-w-3xl mx-auto flex flex-col gap-8">

          {/* Empty state */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-[50vh] text-center opacity-50">
              <span
                className="material-symbols-outlined text-4xl mb-4"
                style={{ fontVariationSettings: "'FILL' 1" }}
              >
                auto_awesome
              </span>
              <h2 className="text-xl font-bold text-gray-900 mb-2">
                How can I help you verify today?
              </h2>
              <p className="text-sm text-gray-500 max-w-md">
                Ask a question, and I will generate an answer while simultaneously verifying it against live web evidence.
              </p>
            </div>
          )}

          {/* Message list */}
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className="flex flex-col gap-6"
              >
                {/* ── User question bubble ─────────────────────────── */}
                <div className="self-end max-w-[85%]">
                  <div className="bg-gray-100 rounded-2xl rounded-tr-sm px-5 py-3.5 text-gray-900 text-[15px] leading-relaxed shadow-sm">
                    {msg.question}
                  </div>
                </div>

                {/* ── AI answer card ───────────────────────────────── */}
                <div
                  onClick={() => onSelectMessage(msg.id)}
                  className={`
                    relative group cursor-pointer transition-all duration-200 border-2 rounded-xl p-5
                    ${selectedMessageId === msg.id
                      ? 'border-[#ff5e5b] bg-[#fffcfc] shadow-[2px_2px_0px_0px_#ff5e5b]'
                      : 'border-transparent hover:border-gray-200 hover:bg-gray-50'}
                  `}
                >
                  {/* AI header row */}
                  <div className="flex items-center gap-2 mb-3 select-none">
                    <div className="w-6 h-6 rounded-md bg-[#111] flex items-center justify-center">
                      <span
                        className="material-symbols-outlined text-white text-[14px]"
                        style={{ fontVariationSettings: "'FILL' 1" }}
                      >
                        smart_toy
                      </span>
                    </div>
                    <span className="text-xs font-bold text-gray-900 uppercase tracking-widest">
                      Mirage
                    </span>

                    {msg.status === 'loading' && (
                      <span className="flex items-center gap-1 ml-2 text-xs font-medium text-gray-500 animate-pulse">
                        Verifying...
                      </span>
                    )}
                  </div>

                  {/* Answer body */}
                  <div className="text-[15px] text-gray-800 leading-relaxed prose prose-sm max-w-none prose-p:my-2 prose-headings:mb-3 prose-headings:mt-4">
                    {msg.status === 'error' ? (
                      <div className="text-red-500 flex items-center gap-2">
                        <span className="material-symbols-outlined">error</span>
                        {msg.errorMessage || 'An error occurred during verification.'}
                      </div>
                    ) : msg.raw_answer ? (
                      <ReactMarkdown>{msg.raw_answer}</ReactMarkdown>
                    ) : (
                      /* Typing indicator dots */
                      <div className="flex items-center gap-2 text-gray-400 py-1">
                        <div
                          className="w-2 h-2 rounded-full bg-gray-300 animate-bounce"
                          style={{ animationDelay: '0ms' }}
                        />
                        <div
                          className="w-2 h-2 rounded-full bg-gray-300 animate-bounce"
                          style={{ animationDelay: '150ms' }}
                        />
                        <div
                          className="w-2 h-2 rounded-full bg-gray-300 animate-bounce"
                          style={{ animationDelay: '300ms' }}
                        />
                      </div>
                    )}
                  </div>

                  {/* Action bar — copy + regenerate (visible on hover / when complete) */}
                  {msg.status === 'complete' && msg.raw_answer && (
                    <div className="flex gap-1 mt-3 pt-3 border-t border-gray-100 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                      <button
                        onClick={(e) => {
                          e.stopPropagation(); // don't select message while copying
                          handleCopy(msg.id, msg.raw_answer!);
                        }}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-gray-400 hover:text-gray-900 hover:bg-gray-100 transition-colors text-xs font-semibold"
                        title="Copy response"
                      >
                        <span className="material-symbols-outlined text-[15px]">
                          {copiedId === msg.id ? 'check' : 'content_copy'}
                        </span>
                        {copiedId === msg.id ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Sentinel — auto-scroll target. A div at the very end of the list. */}
          <div ref={messagesEndRef} aria-hidden />
        </div>
      </div>

      {/* ── Sticky composer ───────────────────────────────────────────── */}
      <div className="sticky bottom-0 px-4 sm:px-8 py-4 bg-gradient-to-t from-white via-white to-transparent">
        <div className="max-w-3xl mx-auto">
          <div className="card p-2 bg-white relative">
            <textarea
              ref={inputRef}
              value={inputVal}
              onChange={(e) => {
                setInputVal(e.target.value);
                resizeTextarea();
              }}
              onKeyDown={handleKeyDown}
              className="w-full bg-transparent resize-none min-h-[52px] max-h-48 px-4 py-3 text-[15px] text-gray-900 placeholder-gray-400 focus:outline-none leading-relaxed"
              placeholder={isLoading ? 'Verifying…' : 'Ask anything…'}
              rows={1}
              disabled={isLoading}
              aria-label="Chat input"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !inputVal.trim()}
              className="absolute right-3 bottom-3 w-8 h-8 flex items-center justify-center rounded-md bg-[#111] text-white hover:bg-[#ff5e5b] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Send message"
            >
              <span
                className="material-symbols-outlined text-[18px]"
                style={{ fontVariationSettings: "'FILL' 1" }}
              >
                arrow_upward
              </span>
            </button>
          </div>
          <p className="text-center text-[11px] text-gray-400 mt-3 uppercase tracking-widest font-semibold">
            Mirage may produce inaccurate results. Verification provides a confidence baseline.
          </p>
        </div>
      </div>
    </div>
  );
}
