// src/lib/traffic-sim.ts

export type CorridorId = "A" | "B" | "C";

export type TrafficNode = {
  id: string;
  corridor: CorridorId;
  value: number; // congestion %
};

export const CORRIDORS = [
  { id: "A", name: "Thika Road" },
  { id: "B", name: "Waiyaki Way" },
  { id: "C", name: "Mombasa Road" },
];

// ✅ corridor name
export function corridorName(id: CorridorId) {
  const corridor = CORRIDORS.find(c => c.id === id);
  return corridor ? corridor.name : "Unknown";
}

// ✅ congestion label
export function congestionLabel(value: number) {
  if (value < 30) return "Low";
  if (value < 70) return "Medium";
  return "High";
}

// ✅ initial nodes
export const initialNodes: TrafficNode[] = [
  { id: "N1", corridor: "A", value: 25 },
  { id: "N2", corridor: "A", value: 60 },
  { id: "N3", corridor: "B", value: 45 },
  { id: "N4", corridor: "B", value: 75 },
  { id: "N5", corridor: "C", value: 30 },
];

// ✅ THIS is what was missing now (tickNode)
export function tickNode(node: TrafficNode): TrafficNode {
  // simulate small random change
  const change = Math.floor(Math.random() * 20 - 10); // -10 to +10
  let newValue = node.value + change;

  // clamp between 0–100
  if (newValue < 0) newValue = 0;
  if (newValue > 100) newValue = 100;

  return {
    ...node,
    value: newValue,
  };
}

// ✅ forecast
export function forecast() {
  return [
    { time: "08:00", value: 20 },
    { time: "09:00", value: 45 },
    { time: "10:00", value: 65 },
    { time: "11:00", value: 80 },
  ];
}