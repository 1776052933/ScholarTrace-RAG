"""根据 PaperScore 对论文去重并重新排序。"""

import re
from typing import Dict, Iterable, List, Mapping, Sequence

from .schema import Paper, PaperScore


class PaperReranker:
    """按论文 ID/标题去重，再按最终分数降序排序。"""

    def deduplicate(self, papers: Iterable[Paper]) -> List[Paper]:
        result, seen_ids, seen_titles = [], set(), set()
        for paper in papers:
            paper_id = paper.paper_id.casefold().strip()
            title = _normalize_title(paper.title)
            if paper_id in seen_ids or title in seen_titles:
                continue
            seen_ids.add(paper_id)
            seen_titles.add(title)
            result.append(paper)
        return result

    def rerank(
        self,
        papers: Sequence[Paper],
        scores: Mapping[str, PaperScore],
    ) -> List[Paper]:
        unique = self.deduplicate(papers)
        missing = [paper.paper_id for paper in unique if paper.paper_id not in scores]
        if missing:
            raise ValueError("missing scores for papers: " + ", ".join(missing))
        return sorted(
            unique,
            key=lambda paper: (
                scores[paper.paper_id].final_score,
                paper.citation_count,
                paper.year or 0,
            ),
            reverse=True,
        )


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title.casefold())
