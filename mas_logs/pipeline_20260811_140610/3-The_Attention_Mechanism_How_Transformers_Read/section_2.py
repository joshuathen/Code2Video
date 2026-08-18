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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["We represent words as three vectors.", "Queries find, Keys match, Values store.", "This is like a library search."]
        self.setup_layout("The Mechanism: Queries, Keys, and Values", lecture_lines)
        
        # Load Assets
        lib_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg", color="#00FF00")
        book_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg", color="#FFFFFF")
        
        # Prepare objects
        q_text = MathTex(r"Q", color="#FFFF00").scale(1.2)
        k_text = MathTex(r"K", color="#FFFF00").scale(1.2)
        v_text = MathTex(r"V", color="#FFFF00").scale(1.2)
        qkv_group = VGroup(q_text, k_text, v_text).arrange(RIGHT, buff=0.5)
        
        mechanism_box = Rectangle(width=4, height=2, color=BLUE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        self.place_in_area(qkv_group, 'B2', 'B4', scale_factor=1.0)
        self.play(FadeIn(qkv_group), FadeIn(lib_icon)) # Adding lib_icon early based on storyboard
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        
        self.place_in_area(mechanism_box, 'B1', 'D5', scale_factor=1.0)
        self.place_at_grid(book_icon, 'C3', scale_factor=0.7)
        
        arrow = Arrow(start=q_text.get_bottom(), end=book_icon.get_top(), color=WHITE)
        match_highlight = SurroundingRectangle(k_text, color="#FF0000")
        self.play(Create(mechanism_box), Create(arrow), Create(match_highlight), FadeIn(book_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        lib_label = Text("Library", font_size=20, color="#00FF00")
        library = VGroup(lib_icon, lib_label).arrange(DOWN)
        self.place_at_grid(library, 'D4', scale_factor=0.8)
        
        self.play(FadeIn(library))
        self.wait(2)
