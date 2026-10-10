import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./style.css";
import "./fatal-error.css";

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    console.error("React render error:", error, info.componentStack);
  }
  render() {
    if (this.state.error) {
      return <main className="fatal-error">
        <h1>The research workspace could not be displayed</h1>
        <p>Refresh the page. If it happens again, share this error:</p>
        <pre>{this.state.error.message}</pre>
      </main>;
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><AppErrorBoundary><App /></AppErrorBoundary></React.StrictMode>
);
