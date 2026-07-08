"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import type { CatalogItem, Recommendation } from "@/lib/api";

type Props = {
  item: CatalogItem | Recommendation;
  draggable?: boolean;
  onDragStart?: (item: CatalogItem) => void;
  onDragEnd?: () => void;
};

export function MovieCard({ item, draggable = false, onDragStart, onDragEnd }: Props) {
  const catalogItem = item as CatalogItem;
  const poster = "poster_url" in item ? item.poster_url : null;
  return (
    <motion.article
      className="poster-tile"
      draggable={draggable}
      onDragStart={() => draggable && onDragStart?.(catalogItem)}
      onDragEnd={onDragEnd}
      whileHover={{ y: -8, rotate: -0.4 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
      style={{ minHeight: 310 }}
    >
      <Link href={`/movie/${item.movie_id}`} aria-label={`Open ${item.title}`}>
        {poster ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={poster} alt="" />
        ) : (
          <div
            style={{
              height: "100%",
              padding: 18,
              display: "grid",
              placeItems: "center",
              fontFamily: "var(--font-editorial-new)",
              fontSize: 38,
              lineHeight: 0.9
            }}
          >
            {item.title}
          </div>
        )}
        <div className="poster-copy">
          <div style={{ fontSize: 11, color: "var(--color-voltage)", letterSpacing: "0.1em" }}>
            {item.release_year ?? "YEAR UNKNOWN"}
          </div>
          <h3 style={{ margin: "6px 0 0", fontSize: 22, lineHeight: 1 }}>{item.title}</h3>
          <p style={{ margin: "8px 0 0", color: "#dfe8df", fontSize: 13 }}>
            {item.genres?.slice(0, 2).join(" / ")}
          </p>
          {"reason" in item ? (
            <p style={{ margin: "10px 0 0", fontSize: 12 }}>{item.reason}</p>
          ) : null}
        </div>
      </Link>
    </motion.article>
  );
}
