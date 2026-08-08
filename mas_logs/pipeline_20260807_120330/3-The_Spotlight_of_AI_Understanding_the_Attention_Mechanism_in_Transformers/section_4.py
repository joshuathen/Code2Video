from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The QKV Framework: The Library Analogy"
        lecture_lines = [
            "Every word generates three vectors: Query, Key, and Value.",
            "The Query represents what the word is looking for.",
            "Keys act like labels on a library's bookshelves.",
            "Values contain the actual information to be retrieved.",
            "Matching Queries to Keys identifies the relevant information."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        QUERY_COLOR = "#00FF00"
        KEY_COLOR = "#3366FF"
        VALUE_COLOR = "#FFCC00"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # A search box labeled 'Query' (#00FF00) appears. Color L1 to #FFFF00.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        query_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.5, color=QUERY_COLOR)
        query_text = Text("Query", font_size=18, color=QUERY_COLOR)
        query_group = VGroup(query_box, query_text)
        
        # [Issue 31 Fix]: query_group at B4 to avoid crowding against title
        self.place_at_grid(query_group, 'B4', scale_factor=0.9)
        
        self.play(FadeIn(query_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Multiple book icons labeled 'Key' (#3366FF) appear. Color L2 to #FFFF00.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        keys_group = VGroup()
        for i in range(3):
            # Book representation
            book_rect = Rectangle(height=0.8, width=0.6, color=KEY_COLOR, fill_opacity=0.3)
            book_spine = Line(book_rect.get_corner(UL), book_rect.get_corner(DL), color=KEY_COLOR, stroke_width=4)
            key_label = Text(f"Key {i+1}", font_size=14, color=KEY_COLOR).next_to(book_rect, DOWN, buff=0.1)
            keys_group.add(VGroup(book_rect, book_spine, key_label))
        
        keys_group.arrange(RIGHT, buff=0.4)
        
        # [Issue 32 Fix]: Use place_in_area to center keys relative to query box
        self.place_in_area(keys_group, 'C3', 'C5', scale_factor=0.8)
        
        self.play(Create(keys_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The 'Value' (#FFCC00) content inside the books is shown. Color L3 to #FFFF00.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        values_group = VGroup()
        for i in range(3):
            val_content = Rectangle(height=0.4, width=0.4, color=VALUE_COLOR, fill_opacity=0.6)
            val_text = Text(f"V{i+1}", font_size=12, color=WHITE).move_to(val_content)
            val_item = VGroup(val_content, val_text)
            # Position values initially inside books
            val_item.move_to(keys_group[i][0].get_center())
            values_group.add(val_item)
        
        self.play(FadeIn(values_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The 'Query' matches with the correct 'Key' book. Color L4 to #FFFF00.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Matching Query to Key 2
        target_key = keys_group[1]
        matching_ray = Line(query_group.get_bottom(), target_key.get_top(), color=HIGHLIGHT_COLOR, stroke_width=2)
        
        self.play(Create(matching_ray))
        self.play(
            target_key.animate.set_color(HIGHLIGHT_COLOR).scale(1.1),
            query_group.animate.set_color(HIGHLIGHT_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The matched book opens, highlighting its 'Value'. Color L5 to #FFFF00.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        target_value = values_group[1]
        others = VGroup(
            keys_group[0], keys_group[2], 
            values_group[0], values_group[2], 
            matching_ray
        )
        
        # Create an isolated pair for the final result
        q_copy = query_group.copy()
        k_copy = target_key.copy()
        v_copy = target_value.copy()
        isolated_pair_group = VGroup(q_copy, k_copy, v_copy).arrange(DOWN, buff=0.5)
        
        # [Issue 33 Fix]: Position isolated pair in area B4-D4 to avoid bottom edge crowding
        self.place_in_area(isolated_pair_group, 'B4', 'D4', scale_factor=0.9)
        
        self.play(
            FadeOut(others),
            FadeOut(query_group),
            FadeOut(target_key),
            FadeOut(target_value),
            FadeIn(isolated_pair_group)
        )
        
        # Highlighting the "Retrieved" value
        retrieved_text = Text("Information Retrieved", font_size=16, color=HIGHLIGHT_COLOR).next_to(isolated_pair_group, DOWN, buff=0.2)
        self.play(
            isolated_pair_group[2].animate.scale(1.3).set_color(HIGHLIGHT_COLOR),
            Write(retrieved_text)
        )
        self.play(Flash(isolated_pair_group[2], color=HIGHLIGHT_COLOR))
        
        self.wait(2)
