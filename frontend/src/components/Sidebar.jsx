import { Link, useLocation } from "react-router-dom";

const navItems = [
    {
        to: "/dashboard",
        label: "Dashboard",
        d: "M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z M9 22V12h6v10",
    },
    {
        to: "/analyze",
        label: "Analyze",
        d: "M22 12h-4l-3 9L9 3l-3 9H2",
    },
    {
        to: "/report",
        label: "Final Report",
        d: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8",
    },
    {
        to: "/dashboard",
        label: "Stats",
        d: "M12 20V10 M6 20V4 M18 20v-6",
    },
];

export default function Sidebar() {
    const location = useLocation();

    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <Link to="/">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2">
                        <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                        <path d="M9 20h6" /><path d="M10 22h4" />
                    </svg>
                </Link>
            </div>
            <nav className="sidebar-nav">
                {navItems.map((item) => (
                    <Link
                        key={item.to + item.label}
                        to={item.to}
                        className={`sidebar-btn ${location.pathname === item.to ? "active" : ""}`}
                        title={item.label}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d={item.d} />
                        </svg>
                    </Link>
                ))}
            </nav>
            <div className="sidebar-bottom">
                <Link to="/" className="sidebar-btn" title="Home">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4 M16 17l5-5-5-5 M21 12H9" />
                    </svg>
                </Link>
            </div>
        </aside>
    );
}
