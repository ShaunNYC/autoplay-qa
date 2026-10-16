import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Self-Evolving Video Game QA",
  description: "An open source game-testing workbench connecting NVIDIA Nemotron, recorded evidence, native replay and developer review.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
