from modules.schema import Link


def group_observations(observations, links: list[Link]):
    """Constrained components: simultaneous detections must remain different fruit."""
    parent = {obs.id: obs.id for obs in observations}
    views = {obs.id: {obs.view_id} for obs in observations}

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for link in sorted(links, key=lambda item: item.score, reverse=True):
        if not link.accepted:
            continue
        if link.a not in parent or link.b not in parent or link.a == link.b:
            raise ValueError("Tautan asosiasi merujuk pengamatan yang tidak valid.")
        a, b = root(link.a), root(link.b)
        if a == b:
            continue
        if views[a] & views[b]:
            link.accepted = False
            continue
        parent[b] = a
        views[a] |= views[b]
    names = {}
    for obs in observations:
        key = root(obs.id)
        names.setdefault(key, f"M{len(names)+1:04d}")
        obs.unique_id = names[key]
    return len(names)


def manual_merge(observations, links, first_id, second_id):
    if first_id == second_id:
        raise ValueError("Pilih dua buah yang berbeda.")
    a = [obs for obs in observations if obs.unique_id == first_id]
    b = [obs for obs in observations if obs.unique_id == second_id]
    if not a or not b or {obs.view_id for obs in a} & {obs.view_id for obs in b}:
        raise ValueError("ID tidak valid atau kedua buah muncul bersamaan dalam satu gambar.")
    links.append(Link(a[0].id, b[0].id, 1.0, "manual-review"))
    return group_observations(observations, links)
