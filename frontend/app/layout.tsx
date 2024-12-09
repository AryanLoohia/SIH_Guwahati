import "./globals.css";
import Nav from "./Navbar";


export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body>
        <Nav></Nav>
        <main>
          <div className="mt-20 overflow-auto custom-scrollbar">{children}</div>
        </main>

      </body>
    </html>
  );
}
