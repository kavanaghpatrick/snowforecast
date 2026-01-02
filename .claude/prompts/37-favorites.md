# Agent Task: Favorites with Local Storage (#37)

## Your Mission
Allow users to save favorite resorts using browser local storage (no account needed).

## Dependencies
```bash
pip install streamlit-local-storage
```

## IMPORTANT: Correct API Usage
The `streamlit-local-storage` package uses `getItem`/`setItem` (not `get`/`set`).
Values must be JSON-serialized strings (not raw lists).

## Files to Create

### `src/snowforecast/dashboard/components/favorites.py`
```python
import json
import streamlit as st
from streamlit_local_storage import LocalStorage

# Initialize storage (must be called in Streamlit context)
def get_storage():
    """Get LocalStorage instance."""
    if 'local_storage' not in st.session_state:
        st.session_state.local_storage = LocalStorage()
    return st.session_state.local_storage

def get_favorites() -> list[str]:
    """Get list of favorite resort names."""
    storage = get_storage()
    raw = storage.getItem("snowforecast_favorites")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []

def save_favorites(favs: list[str]) -> None:
    """Save favorites list to local storage."""
    storage = get_storage()
    storage.setItem(
        itemKey="snowforecast_favorites",
        itemValue=json.dumps(favs)
    )

def add_favorite(resort_name: str) -> None:
    """Add resort to favorites."""
    favs = get_favorites()
    if resort_name not in favs:
        favs.append(resort_name)
        save_favorites(favs)

def remove_favorite(resort_name: str) -> None:
    """Remove resort from favorites."""
    favs = get_favorites()
    if resort_name in favs:
        favs.remove(resort_name)
        save_favorites(favs)

def is_favorite(resort_name: str) -> bool:
    """Check if resort is in favorites."""
    return resort_name in get_favorites()

def render_favorite_toggle(resort_name: str, key_suffix: str = "") -> bool:
    """Render star toggle for favorite. Returns new state."""
    is_fav = is_favorite(resort_name)
    icon = "⭐" if is_fav else "☆"

    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button(icon, key=f"fav_{resort_name}_{key_suffix}", help="Toggle favorite"):
            if is_fav:
                remove_favorite(resort_name)
            else:
                add_favorite(resort_name)
            st.rerun()

    return is_fav

def render_favorites_filter() -> bool:
    """Render 'Show Favorites Only' toggle. Returns filter state."""
    return st.checkbox("⭐ Show Favorites Only", key="filter_favorites")
```

## Integration Points
- Add star toggle to resort cards in detail panel
- Add "Show Favorites Only" filter above map/list
- Favorites persist across browser sessions (localStorage)

## Testing
```python
# Manual test in Streamlit
st.write("Favorites:", get_favorites())
render_favorite_toggle("Alta")
render_favorite_toggle("Snowbird")
if render_favorites_filter():
    st.write("Filtering to favorites only")
```

## Acceptance Criteria
- [ ] Favorites persist across browser sessions
- [ ] Star toggle works (⭐ ↔ ☆)
- [ ] Filter shows only favorites
- [ ] JSON serialization works correctly
- [ ] No account required

## Worktree
Work in: `/Users/patrickkavanagh/snowforecast-worktrees/favorites`
Branch: `phase6/37-favorites`
