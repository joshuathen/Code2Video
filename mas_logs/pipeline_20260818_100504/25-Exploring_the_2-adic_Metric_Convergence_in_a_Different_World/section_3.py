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
        lecture_lines = [
            "2-adic convergence requires differences divisible by high powers.",
            "Partial sums fill bits right to left.",
            "The sequence approaches a 2-adic limit."
        ]
        self.setup_layout("Convergence under 2-adic Metric", lecture_lines)
        
        # Load Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Elements
        powers = [2**i for i in range(1, 5)]
        sequence_group = VGroup(*[Text(f"{p}", color=WHITE, font_size=30) for p in powers]).arrange(RIGHT, buff=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Represent sequence 2, 4, 8... on a line
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(sequence_group, 'A2', 'A5', scale_factor=1.2)
        self.place_at_grid(ruler, 'B2', scale_factor=0.5)
        ruler.set_color("#FFFFFF")
        
        self.play(FadeIn(sequence_group), FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show distances shrinking towards zero
        self.lecture[1].set_color("#FF0000")
        
        arrow_group = VGroup()
        for i in range(len(sequence_group)-1):
            arrow = Arrow(sequence_group[i].get_bottom(), sequence_group[i+1].get_bottom(), color="#FF0000", buff=0.1)
            arrow_group.add(arrow)
            
        self.place_in_area(arrow_group, 'A2', 'A5', scale_factor=1.1)
        self.play(Create(arrow_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display '2-adic limit is 0'
        self.lecture[2].set_color("#FFFF00")
        limit_text = Text("2-adic limit is 0", color="#FFFF00", font_size=36)
        self.place_at_grid(limit_text, 'D3', scale_factor=1.0)
        self.place_at_grid(magnifying_glass, 'D5', scale_factor=0.5)
        magnifying_glass.set_color("#FFFF00")
        
        self.play(Write(limit_text), FadeIn(magnifying_glass))
        self.wait(2)
