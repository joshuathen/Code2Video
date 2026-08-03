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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Mechanics: Query, Key, and Value"
        lines = [
            "The Transformer uses a matching system for every word.",
            "A Query represents what a word is looking for.",
            "The Key describes what that word can offer.",
            "The Value holds the actual information of the word.",
            "Matching Queries to Keys retrieves the most relevant Values."
        ]
        self.setup_layout(title, lines)

        # Colors
        Q_COLOR = "#0000FF"
        K_COLOR = "#FF0000"
        V_COLOR = "#00FF00"
        WORD_COLOR = "#FFFFFF"
        BOOK_COLOR = "#888888"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Word Box - Issue 23: place_in_area('B2', 'C3', 0.7)
        word_box = Rectangle(width=2, height=1.5, color=WORD_COLOR)
        self.place_in_area(word_box, 'B2', 'C3', scale_factor=0.7)
        word_label = Text("Word", font_size=20, color=WORD_COLOR).next_to(word_box, UP, buff=0.1)
        
        # Slots - maintain relative to word_box
        q_slot = Rectangle(width=word_box.width * 0.9, height=word_box.height * 0.25, color=Q_COLOR, fill_opacity=0.3)
        k_slot = Rectangle(width=word_box.width * 0.9, height=word_box.height * 0.25, color=K_COLOR, fill_opacity=0.3)
        v_slot = Rectangle(width=word_box.width * 0.9, height=word_box.height * 0.25, color=V_COLOR, fill_opacity=0.3)
        
        slots = VGroup(q_slot, k_slot, v_slot).arrange(DOWN, buff=0.05).move_to(word_box.get_center())
        q_text = Text("Q", font_size=16, color=Q_COLOR).move_to(q_slot)
        k_text = Text("K", font_size=16, color=K_COLOR).move_to(k_slot)
        v_text = Text("V", font_size=16, color=V_COLOR).move_to(v_slot)
        
        self.play(Create(word_box), Write(word_label))
        self.play(
            FadeIn(q_slot, k_slot, v_slot),
            Write(q_text), Write(k_text), Write(v_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(Q_COLOR)
        
        # Magnifying glass icon
        mag_circle = Circle(radius=0.15, color=Q_COLOR)
        mag_handle = Line(mag_circle.get_bottom(), mag_circle.get_bottom() + [0.1, -0.1, 0], color=Q_COLOR)
        mag_icon = VGroup(mag_circle, mag_handle)
        query_label = Text("Query", font_size=14, color=Q_COLOR).next_to(mag_icon, UP, buff=0.1)
        query_group = VGroup(mag_icon, query_label)
        
        # Issue 22 & 24: place_at_grid('B5', 0.6)
        # First spawn at word box then move
        query_group.move_to(q_slot.get_center()).scale(0.6)
        
        self.play(
            FadeIn(query_group),
            query_group.animate.move_to(self.grid["B5"]),
            run_time=1.5
        )
        
        # Scanning animation
        self.play(
            query_group.animate.move_to(self.grid["B1"]),
            run_time=1,
            rate_func=linear
        )
        self.play(
            query_group.animate.move_to(self.grid["B6"]),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(K_COLOR)
        
        # Row of books
        books = VGroup()
        for i in range(1, 7):
            book_body = Rectangle(width=0.6, height=0.8, color=BOOK_COLOR, fill_opacity=0.5)
            spine_label = Rectangle(width=0.4, height=0.1, color=K_COLOR, fill_opacity=1.0)
            spine_label.move_to(book_body.get_center() + [0, -0.2, 0])
            key_tag = Text("Key", font_size=10, color=K_COLOR).next_to(book_body, DOWN, buff=0.05)
            book = VGroup(book_body, spine_label, key_tag)
            self.place_at_grid(book, f"E{i}", scale_factor=1.0)
            books.add(book)
            
        self.play(LaggedStart(*[FadeIn(b) for b in books], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(V_COLOR)
        
        # Target book (e.g., book 4)
        target_book = books[3]
        target_body = target_book[0]
        
        # "Open" the book - scale it up slightly and reveal value
        value_text = Text("VALUE INFO", font_size=12, color=V_COLOR).move_to(target_body.get_center())
        
        self.play(
            target_book.animate.scale(1.2).set_color(V_COLOR),
            Write(value_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Move Query to Key
        self.play(
            query_group.animate.move_to(target_book.get_top() + [0, 0.3, 0]),
            run_time=1.5
        )
        
        # Flash the match
        match_highlight = Circle(radius=0.3, color=WHITE).move_to(target_book.get_center())
        self.play(Flash(match_highlight))
        
        # Value flows back to Word box
        value_particle = Dot(color=V_COLOR).move_to(target_book.get_center())
        self.add(value_particle)
        
        self.play(
            value_particle.animate.move_to(v_slot.get_center()),
            run_time=2,
            rate_func=bezier([0, 0, 1, 1])
        )
        
        self.play(
            v_slot.animate.set_fill(opacity=0.8),
            FadeOut(value_particle)
        )
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
