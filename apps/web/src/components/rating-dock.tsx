"use client";

import { motion } from "framer-motion";
import type { CatalogItem } from "@/lib/api";

const bins = [
  { label: "Hate", value: 1 },
  { label: "Dislike", value: 2 },
  { label: "Okay", value: 3 },
  { label: "Like", value: 4 },
  { label: "Love", value: 5 }
];

export function RatingDock({
  activeItem,
  onRate
}: {
  activeItem: CatalogItem | null;
  onRate: (rating: number) => void;
}) {
  return (
    <motion.div
      className="rating-dock"
      initial={false}
      animate={{ y: activeItem ? 0 : 140, opacity: activeItem ? 1 : 0 }}
      transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      aria-hidden={!activeItem}
    >
      {bins.map((bin) => (
        <button
          className="rating-bin"
          key={bin.value}
          onDragOver={(event) => event.preventDefault()}
          onDrop={() => onRate(bin.value)}
          data-active={Boolean(activeItem)}
        >
          <strong style={{ display: "block", fontSize: 20 }}>{bin.label}</strong>
          <span style={{ color: "var(--color-sage)", fontSize: 12 }}>{bin.value}.0 rating</span>
        </button>
      ))}
    </motion.div>
  );
}
