from manim import *

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
        self.setup_layout("The Problem: Context Blindness", [
            "Static word embeddings fail to capture polysemous meaning.",
            "Words processed in isolation lose surrounding context.",
            "Imagine a librarian ignoring everything but the first word."
        ])
        
        # Elements
        word_bank = Text("bank", color=YELLOW)
        word_river = Text("river", color=BLUE)
        word_account = Text("account", color=GREEN)
        
        words = VGroup(word_bank, word_river, word_account)
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        librarian = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/librarian.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(word_bank, 'B2', scale_factor=0.8)
        self.place_at_grid(word_river, 'B5', scale_factor=0.8)
        self.place_at_grid(book, 'E3', scale_factor=0.5)
        self.play(FadeIn(word_bank), FadeIn(word_river), FadeIn(book))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color(BLUE))
        self.place_at_grid(word_account, 'C3', scale_factor=0.8)
        self.play(FadeIn(word_account))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color(GREEN))
        
        # Organize for final layout
        word_group = VGroup(word_bank, word_river, word_account)
        self.place_in_area(word_group, 'B2', 'D5', scale_factor=0.9)
        self.place_at_grid(librarian, 'F4', scale_factor=0.6)
        
        line1 = Line(word_bank.get_right(), word_river.get_left(), color=WHITE)
        line2 = Line(word_bank.get_right(), word_account.get_left(), color=WHITE)
        
        self.play(Create(line1), Create(line2), FadeIn(librarian))
        self.wait(2)
