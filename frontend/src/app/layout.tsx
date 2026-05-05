import type { Metadata } from "next";
import { Inter } from "next/font/google"; // Or whatever font you are using
import "./globals.css";
import Providers from "@/components/Providers"; // <-- Import the Provider

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Pulse Traffic | Nairobi Live Corridor Intelligence",
  description: "Live traffic monitoring and predictive analytics for Nairobi corridors.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {/* Wrap your entire app with the TanStack Query Provider */}
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}