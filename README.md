
   ┌───────────────────┐               ┌───────────────────┐
   │   User Features   │               │   Item Features   │
   │ (Age, Gender, Occ)│               │  (Genres, Title)  │
   └─────────┬─────────┘               └─────────┬─────────┘
             │                                   │
             ▼                                   ▼
    ┌─────────────────┐                 ┌─────────────────┐
    │   User Tower    │                 │   Item Tower    │
    │   (2-Layer MLP) │                 │   (2-Layer MLP) │
    └─────────┬─────────┘               └─────────┬─────────┘
              │                                   │
              ▼                                   ▼
      User Vector (32d)                  Item Vector (32d)
              │                                   │
              └───────────────┬───────────────────┘
                              ▼
                 Dot Product / Cosine Sim
                              │
                              ▼
                       Interaction Prob

