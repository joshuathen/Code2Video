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
        lecture_lines = [
            "How do Transformers store and retrieve specific factual knowledge?",
            "Attention handles context, but MLPs act as the memory.",
            "We'll explore how these layers function as knowledge bases."
        ]
        self.setup_layout("The Mystery of Model Memory", lecture_lines)
        
        # Set all lecture lines to gray initially
        for line in self.lecture:
            line.set_color(GRAY)
            
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Text "What is the capital of France?" appears (#FFFFFF) with a question mark icon.
        question_text = Text("What is the capital of France?", font_size=24, color=WHITE)
        self.place_in_area(question_text, "B2", "B5")
        
        q_mark = Text("?", font_size=60, color=WHITE)
        self.place_at_grid(q_mark, "A3")
        
        self.play(Write(question_text), FadeIn(q_mark))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlighting
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(BLUE)
        )
        
        # A "Transformer Block" box appears
        block_box = Rectangle(width=3.5, height=2.5, color=WHITE, stroke_width=2)
        self.place_in_area(block_box, "C2", "E5")
        
        # Fix Issue 29: Use place_in_area for block_label B2-B4
        block_label = Text("Transformer Block", font_size=20, color=WHITE)
        self.place_in_area(block_label, 'B2', 'B4')
        
        # "Attention" highlights in blue (#0000FF)
        attention_text = Text("Attention", font_size=24, color=BLUE)
        self.place_at_grid(attention_text, "C3")
        
        # Words and lines connecting them
        words = VGroup(
            Text("Paris", font_size=18, color=WHITE),
            Text("France", font_size=18, color=WHITE),
            Text("Capital", font_size=18, color=WHITE)
        )
        self.place_at_grid(words[0], "F2")
        self.place_at_grid(words[1], "F4")
        self.place_at_grid(words[2], "F6")
        
        lines = VGroup(*[
            Line(word.get_top(), attention_text.get_bottom(), color=BLUE, stroke_width=1, buff=0.1)
            for word in words
        ])
        
        self.play(
            FadeOut(question_text),
            FadeOut(q_mark),
            Create(block_box),
            Write(block_label),
            Write(attention_text),
            FadeIn(words),
            Create(lines)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlighting
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFD700") # Gold
        )
        
        # Fix Issue 30: "MLP" highlights in gold (#FFD700) at D2
        mlp_text = Text("MLP", font_size=24, color="#FFD700")
        self.place_at_grid(mlp_text, 'D2')
        
        # Fix Issue 25 & 31: Bookshelf icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bookshelf.svg] at D5
        bookshelf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bookshelf.svg")
        bookshelf.set_color(WHITE)
        self.place_at_grid(bookshelf, 'D5', scale_factor=0.6)
        
        self.play(
            attention_text.animate.set_color(DARK_BLUE),
            lines.animate.set_stroke(opacity=0.3),
            Write(mlp_text),
            FadeIn(bookshelf)
        )
        self.wait(2)
