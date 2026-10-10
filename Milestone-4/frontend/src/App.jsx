import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "/api";

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(API + path, {
      credentials: "include",
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
  } catch {
    throw new Error("Could not reach Flask. Check that the backend is running on port 5000.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ("Request failed (" + response.status + ")."));
  return data;
}

function Emblem({ small = false }) {
  return <img className={"emblem" + (small ? " emblem-small" : "")} src="/legalbot-mark.png" alt="" />;
}

function formatInline(text, keyPrefix) {
  return String(text).split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={keyPrefix + "-" + index}>{part.slice(2, -2)}</strong>
      : <span key={keyPrefix + "-" + index}>{part}</span>
  );
}

function formatAnswer(content) {
  return String(content || "").split(/\r?\n/).map((raw, index) => {
    const line = raw.trim();
    if (!line) return <div className="answer-space" key={index} />;
    const heading = line.match(/^#{0,3}\s*(Relevant legal provisions|Legal summary|Citations\s*&\s*sources|Legal disclaimer)\s*:?\s*$/i);
    if (heading) {
      return <h3 className="answer-heading" key={index}>
        <span>§</span>{heading[1].toUpperCase()}
      </h3>;
    }
    const excerpt = line.match(/^>\s?(.*)$/);
    if (excerpt) return <blockquote key={index}>{formatInline(excerpt[1], "q-" + index)}</blockquote>;
    const bullet = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    if (bullet) return <div className="answer-bullet" key={index}><span>•</span><div>{formatInline(bullet[1], "b-" + index)}</div></div>;
    const source = line.match(/^Source:\s*(.*)$/i);
    if (source) return <div className="source-row" key={index}><small>Source</small>{formatInline(source[1], "s-" + index)}</div>;
    return <p className="answer-paragraph" key={index}>{formatInline(line, "p-" + index)}</p>;
  });
}

function Brand() {
  return <div className="brand">
    <Emblem />
    <div><strong>Legal Reference</strong><small>Case retrieval system</small></div>
  </div>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [chats, setChats] = useState([]);
  const [active, setActive] = useState(null);
  const [messages, setMessages] = useState([]);
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [mobileNav, setMobileNav] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [profilePhoto, setProfilePhoto] = useState("");
  const [profilePhotoChanged, setProfilePhotoChanged] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);
  const [deletingChat, setDeletingChat] = useState(false);
  const chatWindowRef = useRef(null);
  const inputRef = useRef(null);
  const photoInputRef = useRef(null);

  async function refreshChats() {
    const data = await api("/conversations");
    const rows = data.conversations || [];
    setChats(rows);
    return rows;
  }

  useEffect(() => {
    let cancelled = false;
    async function restore() {
      try {
        const data = await api("/auth/me");
        if (cancelled) return;
        setUser(data.user);
        const rows = await refreshChats();
        if (cancelled || !rows.length) return;
        const chat = await api("/conversations/" + rows[0].id);
        if (!cancelled) {
          setActive(chat.conversation.id);
          setMessages(chat.conversation.messages || []);
        }
      } catch (err) {
        if (!cancelled && err.message.includes("Could not reach Flask")) setError(err.message);
      }
    }
    async function checkHealth() {
      try {
        const data = await api("/health");
        if (!cancelled) setHealth(data);
      } catch {
        if (!cancelled) setHealth(null);
      }
    }
    void restore();
    void checkHealth();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const element = chatWindowRef.current;
    if (element) element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function submitAuth(event) {
    event.preventDefault();
    setError("");
    if (mode === "signup" && form.password !== form.confirm) {
      setError("The passwords do not match.");
      return;
    }
    try {
      const endpoint = mode === "login" ? "/auth/login" : "/auth/signup";
      const result = await api(endpoint, {
        method: "POST",
        body: JSON.stringify({
          name: form.name.trim(),
          email: form.email.trim(),
          password: form.password,
        }),
      });
      setUser(result.user);
      setForm({ name: "", email: "", password: "", confirm: "" });
      await refreshChats();
    } catch (err) {
      setError(err.message);
    }
  }

  async function startChat() {
    setError("");
    setMobileNav(false);
    try {
      const result = await api("/conversations", { method: "POST", body: "{}" });
      setActive(result.conversation.id);
      setMessages([]);
      await refreshChats();
      inputRef.current?.focus();
    } catch (err) {
      setError(err.message);
    }
  }

  async function openChat(id) {
    setError("");
    setMobileNav(false);
    try {
      const result = await api("/conversations/" + id);
      setActive(result.conversation.id);
      setMessages(result.conversation.messages || []);
    } catch (err) {
      setError(err.message);
    }
  }

  function openProfile() {
    setProfileName(user.name || "");
    setProfilePhoto(user.photo || "");
    setProfilePhotoChanged(false);
    setProfileError("");
    setProfileOpen(true);
  }

  function selectProfilePhoto(event) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      setProfileError("Choose a PNG, JPEG, or WebP image.");
      return;
    }
    if (file.size > 1_000_000) {
      setProfileError("Profile photos must be 1 MB or smaller.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        setProfileError("The selected profile photo could not be read.");
        return;
      }
      setProfilePhoto(reader.result);
      setProfilePhotoChanged(true);
      setProfileError("");
    };
    reader.onerror = () => setProfileError("The selected profile photo could not be read.");
    reader.readAsDataURL(file);
  }

  async function saveProfile(event) {
    event.preventDefault();
    const name = profileName.trim();
    if (!name || name.length > 120) {
      setProfileError("Enter a name of 1 to 120 characters.");
      return;
    }
    setSavingProfile(true);
    setProfileError("");
    const payload = { name };
    if (profilePhotoChanged) {
      if (profilePhoto) payload.photo = profilePhoto;
      else payload.remove_photo = true;
    }
    try {
      const result = await api("/auth/profile", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setUser(result.user);
      setProfileOpen(false);
    } catch (err) {
      setProfileError(err.message);
    } finally {
      setSavingProfile(false);
    }
  }

  async function deleteChat() {
    if (!chatToDelete || deletingChat) return;
    const deletedId = chatToDelete.id;
    setDeletingChat(true);
    setError("");
    try {
      await api("/conversations/" + deletedId, { method: "DELETE" });
      const rows = await refreshChats();
      setChatToDelete(null);
      if (active === deletedId) {
        if (rows.length) {
          const result = await api("/conversations/" + rows[0].id);
          setActive(result.conversation.id);
          setMessages(result.conversation.messages || []);
        } else {
          setActive(null);
          setMessages([]);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingChat(false);
    }
  }

  async function submitQuery(event) {
    event.preventDefault();
    const question = query.trim();
    if (!question || busy) return;
    setQuery("");
    setError("");
    setBusy(true);
    setMessages((current) => [...current, { role: "user", content: question }]);
    try {
      let conversationId = active;
      if (!conversationId) {
        const created = await api("/conversations", { method: "POST", body: "{}" });
        conversationId = created.conversation.id;
        setActive(conversationId);
      }
      const result = await api("/conversations/" + conversationId + "/messages", {
        method: "POST",
        body: JSON.stringify({ message: question }),
      });
      setMessages((current) => [...current, { role: "assistant", content: result.answer }]);
      await refreshChats();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    try {
      await api("/auth/logout", { method: "POST", body: "{}" });
    } finally {
      setUser(null);
      setChats([]);
      setActive(null);
      setMessages([]);
      setQuery("");
      setError("");
    }
  }

  if (!user) {
    return <main className="auth-layout">
      <section className="auth-story">
        <Brand />
        <div className="story-copy">
          <p className="eyebrow"><span /> INDIAN LAW · RESEARCH · CASES</p>
          <h1>AI-Based Legal Reference and Case Retrieval System</h1>
          <p className="story-intro">Ask questions about Indian law and get answers grounded in relevant passages from indexed statutes and landmark judgments.</p>
          <div className="story-features">
            <div><span>⌕</span><p><strong>Search Indian legal sources</strong><small>Find relevant passages from indexed Acts and judgments.</small></p></div>
            <div><span>§</span><p><strong>Read the full context</strong><small>Review the provision and its surrounding text.</small></p></div>
            <div><span>✦</span><p><strong>Research with local AI</strong><small>Ollama generation with Pinecone retrieval.</small></p></div>
          </div>
        </div>
        <p className="story-note">For learning and legal research. Not a substitute for professional legal advice.</p>
      </section>
      <section className="auth-panel">
        <form className="auth-card" onSubmit={submitAuth}>
          <div className="auth-brand"><Emblem small /><span>AI Legal Reference</span></div>
          <p className="eyebrow">YOUR PRIVATE RESEARCH WORKSPACE</p>
          <h2>{mode === "login" ? "Welcome back" : "Create an account"}</h2>
          <p className="auth-help">{mode === "login" ? "Sign in to continue your legal research." : "Save your questions and conversations in one place."}</p>
          <div className="auth-switch">
            <button type="button" className={mode === "login" ? "chosen" : ""} onClick={() => { setMode("login"); setError(""); }}>Sign in</button>
            <button type="button" className={mode === "signup" ? "chosen" : ""} onClick={() => { setMode("signup"); setError(""); }}>Create account</button>
          </div>
          {error && <div className="notice error-notice" role="alert">{error}</div>}
          {mode === "signup" && <label>Full name<input required autoComplete="name" placeholder="Your name" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>}
          <label>Email address<input required type="email" autoComplete="email" placeholder="name@example.com" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} /></label>
          <label>Password<input required minLength={8} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} placeholder="At least 8 characters" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} /></label>
          {mode === "signup" && <label>Confirm password<input required minLength={8} type="password" autoComplete="new-password" placeholder="Enter it once more" value={form.confirm} onChange={(event) => setForm({ ...form, confirm: event.target.value })} /></label>}
          <button className="auth-submit" type="submit">{mode === "login" ? "Sign in to workspace" : "Create account"}<span>→</span></button>
          <p className="auth-private"><span>●</span> Your account and conversations are private to you.</p>
        </form>
        <p className="auth-caption">AI-Based Legal Reference and Case Retrieval System</p>
      </section>
    </main>;
  }

  return <main className="workspace">
    <button className={"nav-scrim" + (mobileNav ? " visible" : "")} aria-label="Close navigation" onClick={() => setMobileNav(false)} />
    <aside className={"sidebar" + (mobileNav ? " sidebar-open" : "")}>
      <Brand />
      <button className="new-chat" onClick={startChat}><span>＋</span> New research chat</button>
      <div className="library-label"><span>YOUR LIBRARY</span><small>{chats.length}</small></div>
      <nav className="chat-list" aria-label="Conversation history">
        {chats.map((chat) => <div key={chat.id} className={"chat-row" + (active === chat.id ? " active" : "")}>
          <button className="chat-link" onClick={() => openChat(chat.id)} title={chat.title}>
            <span className="search-glyph">⌕</span><span>{chat.title}</span>
          </button>
          <button className="chat-delete" onClick={() => setChatToDelete(chat)} title="Delete chat" aria-label={"Delete chat: " + chat.title}>
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11m-9.5 0 .7 10h7.6L15 6M8 6V4h4v2m-3 3v4m2-4v4" /></svg>
          </button>
        </div>)}
        {!chats.length && <div className="empty-library"><Emblem small /><p>Your research conversations will appear here.</p></div>}
      </nav>
      <div className="sidebar-user">
        <span className="user-avatar">{user.photo ? <img src={user.photo} alt="" /> : (user.name?.slice(0, 1)?.toUpperCase() || "U")}</span>
        <div className="user-details"><strong>{user.name}</strong><small title={user.email}>{user.email}</small></div>
        <button className="logout profile-settings" onClick={openProfile} title="Profile settings" aria-label="Profile settings">⚙</button>
        <button className="logout" onClick={signOut} title="Sign out" aria-label="Sign out">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M10 17l5-5-5-5m5 5H3" />
            <path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" />
          </svg>
        </button>
      </div>
    </aside>

    <section className="chat-panel">
      <header className="topbar">
        <button className="mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation">☰</button>
        <div className="topbar-title"><span>RESEARCH WORKSPACE</span><strong>AI-Based Legal Reference and Case Retrieval System</strong></div>
        <div className="connection-status" title={health?.ollama_available ? "Ollama is responding" : "Ollama status unavailable"}>
          <span className={"health-dot" + (health?.ollama_available ? " online" : "")} />
          <span>Local AI</span><i /><span>Pinecone retrieval</span>
        </div>
      </header>

      <section ref={chatWindowRef} className={"conversation-view" + (messages.length ? " populated" : "")} aria-live="polite">
        {!messages.length && <div className="welcome">
          <div className="welcome-emblem"><Emblem /></div>
          <p className="eyebrow">INDIAN LEGAL RESEARCH ASSISTANT</p>
          <h1>What are you researching today?</h1>
          <p>Ask about a provision, an Act, or a case in the indexed legal sources.</p>
          <div className="prompt-title">START WITH A QUESTION</div>
          <div className="prompt-suggestions">
            {[
              "Explain Article 21 of the Constitution",
              "What did Kesavananda Bharati decide?",
              "Summarize the Right to Information Act",
            ].map((suggestion) => <button key={suggestion} onClick={() => { setQuery(suggestion); inputRef.current?.focus(); }}>
              <span>{suggestion}</span><i>↗</i>
            </button>)}
          </div>
        </div>}
        {messages.map((message, index) => <article className={"message-row " + (message.role === "user" ? "from-user" : "from-assistant")} key={message.id || (index + "-" + message.role)}>
          <div className={"message-avatar" + (message.role === "user" ? " person" : "")}>
            {message.role === "user"
              ? (user.photo ? <img src={user.photo} alt="" /> : (user.name?.slice(0, 1)?.toUpperCase() || "U"))
                : <Emblem />}
          </div>
          <div className="message-body">
            <strong className="message-author">{message.role === "user" ? "YOU" : "LEGAL RESEARCH ASSISTANT"}</strong>
            <div className="formatted-answer">{formatAnswer(message.content)}</div>
          </div>
        </article>)}
        {busy && <article className="message-row from-assistant">
          <div className="message-avatar assistant-avatar"><Emblem /></div>
          <div className="message-body"><strong className="message-author">LEGAL RESEARCH ASSISTANT</strong>
            <div className="typing-dots" aria-label="Searching legal documents"><span /><span /><span /></div>
            <small className="thinking">Searching your sources and preparing a response…</small>
          </div>
        </article>}
      </section>

      <footer className="composer-area">
        {error && <div className="notice request-notice" role="alert">{error}</div>}
        <form className="composer" onSubmit={submitQuery}>
          <textarea ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submitQuery(event); } }}
            rows={1} placeholder="Ask a question about Indian law…" aria-label="Legal question" />
          <button type="submit" disabled={!query.trim() || busy} aria-label="Send question">
            <svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 11 12 6l5 5M12 7v11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
        </form>
        <p className="legal-note">For legal research only. Verify sources and consult a qualified lawyer for advice.</p>
      </footer>
    </section>

    {profileOpen && <div className="modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !savingProfile) setProfileOpen(false);
    }}>
      <section className="app-modal profile-modal" role="dialog" aria-modal="true" aria-labelledby="profile-title">
        <button className="modal-close" type="button" onClick={() => setProfileOpen(false)} aria-label="Close profile settings" disabled={savingProfile}>×</button>
        <p className="eyebrow">ACCOUNT SETTINGS</p>
        <h2 id="profile-title">Your profile</h2>
        <form onSubmit={saveProfile}>
          <div className="profile-photo-row">
            <span className="profile-preview">{profilePhoto ? <img src={profilePhoto} alt="" /> : (profileName.trim().slice(0, 1).toUpperCase() || "U")}</span>
            <div className="profile-photo-actions">
              <strong>Profile photo</strong>
              <button type="button" className="secondary-button" onClick={() => photoInputRef.current?.click()}>Upload photo</button>
              {profilePhoto && <button type="button" className="text-button" onClick={() => {
                setProfilePhoto("");
                setProfilePhotoChanged(true);
                setProfileError("");
              }}>Remove photo</button>}
              <input ref={photoInputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={selectProfilePhoto} />
              <small>PNG, JPEG, or WebP · up to 1 MB</small>
            </div>
          </div>
          <label className="profile-name-label">Display name
            <input autoComplete="name" maxLength={120} value={profileName} onChange={(event) => setProfileName(event.target.value)} />
          </label>
          {profileError && <div className="notice error-notice" role="alert">{profileError}</div>}
          <div className="modal-actions">
            <button type="button" className="secondary-button" onClick={() => setProfileOpen(false)} disabled={savingProfile}>Cancel</button>
            <button type="submit" className="primary-button" disabled={savingProfile}>{savingProfile ? "Saving…" : "Save changes"}</button>
          </div>
        </form>
      </section>
    </div>}

    {chatToDelete && <div className="modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !deletingChat) setChatToDelete(null);
    }}>
      <section className="app-modal delete-modal" role="dialog" aria-modal="true" aria-labelledby="delete-chat-title">
        <div className="delete-icon"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11m-9.5 0 .7 10h7.6L15 6M8 6V4h4v2m-3 3v4m2-4v4" /></svg></div>
        <h2 id="delete-chat-title">Delete this chat?</h2>
        <p><strong>{chatToDelete.title}</strong> and all of its messages will be permanently deleted.</p>
        <div className="modal-actions">
          <button type="button" className="secondary-button" onClick={() => setChatToDelete(null)} disabled={deletingChat}>Cancel</button>
          <button type="button" className="danger-button" onClick={deleteChat} disabled={deletingChat}>{deletingChat ? "Deleting…" : "Delete chat"}</button>
        </div>
      </section>
    </div>}
  </main>;
}
