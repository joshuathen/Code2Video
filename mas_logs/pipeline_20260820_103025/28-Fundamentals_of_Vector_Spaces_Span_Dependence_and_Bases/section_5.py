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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Span defines the reach of your system.",
            "Linear independence prevents all redundant information.",
            "Basis provides the most efficient coordinate description."
        ]
        self.setup_layout("Summary & Application", lecture_lines)
        
        # Define visual elements
        # 1. Span: Map icon
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        map_icon.set_color("#FF5733")
        span_label = Text("Span", font_size=20)
        span_label.next_to(map_icon, UP, buff=0.1)
        span_group = VGroup(map_icon, span_label)

        # 2. Linear Independence: Unique Components
        indep_icon = Circle(radius=0.5, color="#33FF57")
        indep_label = Text("Indep.", font_size=20)
        indep_label.next_to(indep_icon, UP, buff=0.1)
        indep_group = VGroup(indep_icon, indep_label)

        # 3. Basis: Grid icon
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid_icon.set_color("#3357FF")
        basis_label = Text("Basis", font_size=20)
        basis_label.next_to(grid_icon, UP, buff=0.1)
        basis_group = VGroup(grid_icon, basis_label)

        # Layout groups as per critical feedback (B004, B019, VideoCritic)
        # Using segments Row B-C and D-F for distinct components
        self.place_in_area(span_group, 'B4', 'C6', scale_factor=0.7)
        self.place_in_area(indep_group, 'D4', 'E6', scale_factor=0.7)
        # Place basis below others
        self.place_in_area(basis_group, 'E4', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(span_group))
        self.lecture[0].set_color("#FF5733")
        self.play(Indicate(map_icon))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(indep_group))
        self.lecture[1].set_color("#33FF57")
        self.play(Flash(indep_icon))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(basis_group))
        self.lecture[2].set_color("#3357FF")
        
        basis_efficiency_text = Text("Basis = Efficiency", color=GOLD)
        # Placement fix from issue #34
        self.place_at_grid(basis_efficiency_text, 'F4', scale_factor=0.8)
        self.play(Write(basis_efficiency_text))
        
        self.wait(2)
