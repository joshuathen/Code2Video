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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data
        title_text = "The Mystery of Model Memory"
        lecture_lines = [
            "Attention connects words, but where are facts stored?",
            "Transformers use MLP layers as a vast knowledge vault.",
            "How does the model remember that Paris is in France?"
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)
        self.title.set_color("#ADD8E6")
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Book icon for "Context" (Attention mechanism's focus)
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg]
        book_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg", color="#ADD8E6", height=1.5)
        book_label = Text("Context", font_size=20, color="#ADD8E6")
        book_group = VGroup(book_icon, book_label).arrange(DOWN, buff=0.1)
        
        # Fix Issue 30: Use area B3 to F6
        self.place_in_area(book_group, "B3", "F6", scale_factor=1.0)
        
        self.play(FadeIn(book_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transition lecture line focus
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Vast background grid for MLP (The Knowledge Vault)
        vault_slots = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                slot = Square(side_length=0.8, stroke_width=1, stroke_color=GREY_C, fill_opacity=0.1)
                self.place_at_grid(slot, f"{r}{c}")
                vault_slots.add(slot)
        
        vault_label = Text("MLP: The Knowledge Store", font_size=24, color=GOLD_A)
        # Fix Issue 28: Position at A4-A6
        self.place_in_area(vault_label, 'A4', 'A6', scale_factor=1.0)

        self.play(
            book_group.animate.scale(0.3).move_to(self.grid["F6"]), # Move "Context" further to the side and smaller
            FadeIn(vault_slots, lag_ratio=0.01),
            Write(vault_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Transition lecture line focus
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlight a specific fact inside the vault: "Paris is in France"
        # Fix Issue 29: Use C4 and scale 0.8
        target_slot = vault_slots[15] # C4 index = 2*6 + 3 = 15
        highlight_rect = target_slot.copy().set_color(YELLOW).set_stroke(width=4)
        
        fact_text = Text("Paris -> France", font_size=18, color=YELLOW)
        self.place_at_grid(fact_text, 'C4', scale_factor=0.8)
        
        self.play(
            Create(highlight_rect),
            Write(fact_text)
        )
        self.wait(3)
