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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "A basis must span the entire space. [Asset: SpanningSet]",
            "A basis must be linearly independent. [Asset: IndependentSet]",
            "Every point has one unique representation. [Asset: UniqueCoord]"
        ]
        self.setup_layout("Bases: The Perfect Coordinate System", lecture_lines)
        
        # Elements
        basis_v1 = Arrow(start=ORIGIN, end=RIGHT*1.5, color="#FF00FF")
        basis_v2 = Arrow(start=ORIGIN, end=UP*1.5, color="#FF00FF")
        basis_group = VGroup(basis_v1, basis_v2)
        
        grid_lines = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        dot = Dot(color="#FFFFFF")
        
        animation_container = VGroup(basis_group, grid_lines, dot)
        
        # Apply layout fixes from VideoCritic/Orchestrator
        self.place_at_grid(basis_group, 'C5', scale_factor=0.6)
        self.place_at_grid(grid_lines, 'C5', scale_factor=0.4)
        self.place_in_area(animation_container, 'D4', 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"), Create(basis_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#808080"), FadeIn(grid_lines))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"), FadeIn(dot))
        self.wait(2)
