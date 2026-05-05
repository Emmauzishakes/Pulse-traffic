export interface BackendTrafficReading {
  id: number;
  node_id: string;
  timestamp: string;
  vehicle_count: number;
  speed: number;
  density: number;
}

export interface TrafficNodeMeta {
  id: string;
  name: string;
  corridor: string;
  lat: number;
  lng: number;
}

export interface EnrichedTrafficData extends BackendTrafficReading {
  meta: TrafficNodeMeta;
  status: 'Free' | 'Slow' | 'Jammed';
}

// Static mapping to bridge the gap until the backend has a /nodes endpoint
export const NODE_DIRECTORY: Record<string, TrafficNodeMeta> = {
  // --- Thika Superhighway (A2) ---
  'ST-THIKA-001': { id: 'ST-THIKA-001', name: 'Globe Roundabout (CBD)', corridor: 'Thika Superhighway', lat: -1.2780, lng: 36.8180 },
  'ST-THIKA-002': { id: 'ST-THIKA-002', name: 'Pangani Interchange', corridor: 'Thika Superhighway', lat: -1.2650, lng: 36.8350 },
  'ST-THIKA-003': { id: 'ST-THIKA-003', name: 'Muthaiga Roundabout', corridor: 'Thika Superhighway', lat: -1.2580, lng: 36.8400 },
  'ST-THIKA-004': { id: 'ST-THIKA-004', name: 'Kasarani Interchange', corridor: 'Thika Superhighway', lat: -1.2200, lng: 36.8800 },
  'ST-THIKA-005': { id: 'ST-THIKA-005', name: 'Juja Flyover / JKUAT', corridor: 'Thika Superhighway', lat: -1.1030, lng: 37.0144 },

  // --- Nairobi Expressway & Mombasa Rd ---
  'ST-EXP-001': { id: 'ST-EXP-001', name: 'Westlands Toll Station', corridor: 'Nairobi Expressway', lat: -1.2660, lng: 36.8000 },
  'ST-EXP-002': { id: 'ST-EXP-002', name: 'Museum Hill Toll', corridor: 'Nairobi Expressway', lat: -1.2730, lng: 36.8130 },
  'ST-EXP-003': { id: 'ST-EXP-003', name: 'Capital Centre', corridor: 'Mombasa Road', lat: -1.3050, lng: 36.8350 },
  'ST-EXP-004': { id: 'ST-EXP-004', name: 'JKIA Turnoff', corridor: 'Nairobi Expressway', lat: -1.3340, lng: 36.9060 },
  'ST-EXP-005': { id: 'ST-EXP-005', name: 'Mlolongo Toll Station', corridor: 'Nairobi Expressway', lat: -1.3850, lng: 36.9380 },

  // --- Ngong Road ---
  'ST-NGONG-001': { id: 'ST-NGONG-001', name: 'Prestige Plaza', corridor: 'Ngong Road', lat: -1.2990, lng: 36.7860 },
  'ST-NGONG-002': { id: 'ST-NGONG-002', name: 'Junction Mall', corridor: 'Ngong Road', lat: -1.2980, lng: 36.7620 },
  'ST-NGONG-003': { id: 'ST-NGONG-003', name: 'Karen Roundabout', corridor: 'Ngong Road', lat: -1.3190, lng: 36.7050 },

  // --- Lang'ata Road & Environs ---
  'ST-LANG-001': { id: 'ST-LANG-001', name: 'T-Mall Roundabout', corridor: 'Lang\'ata Road', lat: -1.3130, lng: 36.8140 },
  'ST-LANG-002': { id: 'ST-LANG-002', name: 'Bomas of Kenya', corridor: 'Lang\'ata Road', lat: -1.3400, lng: 36.7660 },
};