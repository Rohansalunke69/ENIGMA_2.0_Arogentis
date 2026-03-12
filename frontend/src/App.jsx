import { BrowserRouter, Routes, Route } from "react-router-dom";
// import Landing from "./pages/Landing"; // Original landing (preserved)
import CinematicLanding from "./pages/CinematicLanding";
import Dashboard from "./pages/Dashboard";
import Analyze from "./pages/Analyze";
import Report from "./pages/Report";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CinematicLanding />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/report" element={<Report />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
