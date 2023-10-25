import textwrap
import pandas as pd


class EssayChunksDivider:
    def get_chunks_df(self, essays_df: pd.DataFrame, chunk_size: int):
        chunks_df_rows = []

        for index, row in essays_df.iterrows():
            essay_id = row["essay_id"]
            essay_text = row["essay_text"]

            chunks = self._split_text_into_chunks(essay_text, chunk_size)

            for chunk in chunks:
                new_row = {
                    "essay_id": essay_id,
                    "essay_text": chunk,
                    "second_grade": row["second_grade"],
                }
                chunks_df_rows.append(new_row)

        chunks_df = (
            pd.DataFrame(chunks_df_rows)
            .sort_values(by=["second_grade", "essay_id"])
            .reset_index(drop=True)
        )

        return chunks_df

    def _split_text_into_chunks(self, text, chunk_size):
        return textwrap.wrap(text, chunk_size)
