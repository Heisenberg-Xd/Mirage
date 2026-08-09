import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Workspace from './pages/Workspace';
import NotFound from './pages/NotFound';
import { ToastProvider } from './hooks/useToast';
import ToastContainer from './components/ToastContainer';
import type { ConversationMessage } from './types';

function App() {
  // Conversation state lives at App level so it persists across route changes.
  // Navigating /chat → / → /chat keeps the full conversation history intact.
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);

  return (
    <ToastProvider>
      <Router>
        <Routes>
          {/* ── Marketing landing page ─────────────────────────────── */}
          <Route path="/" element={<LandingPage />} />

          {/* ── Chat workspace ─────────────────────────────────────── */}
          <Route
            path="/chat"
            element={
              <Workspace
                messages={messages}
                setMessages={setMessages}
                selectedMessageId={selectedMessageId}
                setSelectedMessageId={setSelectedMessageId}
              />
            }
          />

          {/* ── 404 — any unknown route ────────────────────────────── */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
      <ToastContainer />
    </ToastProvider>
  );
}

export default App;
