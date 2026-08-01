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

class Section2PrerequisitesScene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            'Traditional geometry measures fixed distances and angles.',
            'Topology ignores size, focusing only on connectivity.',
            'A circle and a square are topologically the same.'
        ]
        
        self.setup_layout("Prerequisite: Euclidean vs. Topological Thinking", lecture_lines)

        # Colors
        COLOR_WOOD = "#8B4513"
        COLOR_RUBBER = "#00FF00"
        COLOR_WHITE = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Euclidean wooden square - Updated position/scale per Issue 57
        wood_square = Square(side_length=2.0, color=COLOR_WOOD, fill_opacity=0.6)
        euclidean_label = Text("Euclidean (Rigid)", font_size=18, color=COLOR_WOOD)
        
        # Topological rubber square (initially a square)
        rubber_square = Square(side_length=2.0, color=COLOR_RUBBER, fill_opacity=0.3)
        topo_label = Text("Topological (Flexible)", font_size=18, color=COLOR_RUBBER)

        # Position them - Updated per Issue 57
        self.place_in_area(wood_square, "B2", "D4", scale_factor=0.8)
        self.place_in_area(euclidean_label, "A2", "A3")
        
        self.place_in_area(rubber_square, "B4", "D6")
        self.place_in_area(topo_label, "A4", "A6")

        # Visual markers for "distances and angles"
        dist_line = Line(wood_square.get_corner(DL), wood_square.get_corner(DR), color=COLOR_WHITE).shift(DOWN*0.2)
        dist_label = Text("Fixed Length", font_size=14, color=COLOR_WHITE).next_to(dist_line, DOWN, buff=0.1)
        angle_arc = Arc(radius=0.3, start_angle=0, angle=PI/2, color=COLOR_WHITE).move_to(wood_square.get_corner(DL), aligned_edge=DL)
        angle_label = Text("90°", font_size=14, color=COLOR_WHITE).next_to(angle_arc, UR, buff=0.05)

        self.play(
            FadeIn(wood_square),
            FadeIn(euclidean_label),
            FadeIn(rubber_square),
            FadeIn(topo_label)
        )
        self.play(Create(dist_line), Write(dist_label), Create(angle_arc), Write(angle_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Morph rubber square into a circle
        target_circle = Circle(radius=1.0, color=COLOR_RUBBER, fill_opacity=0.3)
        self.place_in_area(target_circle, "B4", "D6")
        
        # Wooden square stays rigid - small shake to emphasize rigidity
        self.play(
            ReplacementTransform(rubber_square, target_circle),
            wood_square.animate.scale(1.05).set_rate_func(there_and_back),
            FadeOut(dist_line), FadeOut(dist_label), FadeOut(angle_arc), FadeOut(angle_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Dash lines around both to show boundary
        wood_boundary = DashedVMobject(Square(side_length=2.2, color=COLOR_WHITE))
        circle_boundary = DashedVMobject(Circle(radius=1.1, color=COLOR_WHITE))
        
        # Position boundaries to match current positions of square/circle
        self.place_in_area(wood_boundary, "B2", "D4", scale_factor=0.8) # Adjusted to match wood_square
        self.place_in_area(circle_boundary, "B4", "D6")
        
        label_boundary_1 = Text("One Boundary", font_size=16, color=COLOR_WHITE)
        label_boundary_2 = Text("One Boundary", font_size=16, color=COLOR_WHITE)
        
        # Updated positioning per Issue 57
        self.place_at_grid(label_boundary_1, "D2", scale_factor=0.8)
        self.place_at_grid(label_boundary_2, "D5", scale_factor=0.8)

        self.play(
            Create(wood_boundary),
            Create(circle_boundary),
            Write(label_boundary_1),
            Write(label_boundary_2)
        )
        self.wait(2)

        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
