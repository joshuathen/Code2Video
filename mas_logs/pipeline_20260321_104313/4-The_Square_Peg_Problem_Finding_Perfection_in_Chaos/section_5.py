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
        # Title and lecture lines setup
        title_text = "The 3D Transformation: The Configuration Space"
        lecture_lines = [
            "Pick two points, A and B, on our loop.",
            "Measure the distance between these two points.",
            "Map this distance onto a square parameter grid.",
            "Color each grid point based on that distance.",
            "This creates a surface representing all point pairs."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Hex colors cast to ManimColor to ensure they have the .interpolate method
        COLOR_LOOP = ManimColor("#FFFFFF")      # White
        COLOR_POINTS = ManimColor("#FFFF00")    # Yellow
        COLOR_DISTANCE = ManimColor("#00FFFF")  # Cyan
        COLOR_SURFACE = ManimColor("#FF00FF")   # Magenta

        # --- Prepare Mobjects ---
        
        # Loop and points setup
        loop = Circle(radius=1.2, color=COLOR_LOOP, stroke_width=4)
        p_a_rel = loop.point_at_angle(45 * DEGREES)
        p_b_rel = loop.point_at_angle(165 * DEGREES)
        dot_a = Dot(p_a_rel, color=COLOR_POINTS, radius=0.08)
        dot_b = Dot(p_b_rel, color=COLOR_POINTS, radius=0.08)
        label_a = Text("A", font_size=20, color=COLOR_POINTS).next_to(dot_a, UR, buff=0.1)
        label_b = Text("B", font_size=20, color=COLOR_POINTS).next_to(dot_b, UL, buff=0.1)
        
        loop_group = VGroup(loop, dot_a, dot_b, label_a, label_b)
        # Position loop group according to Issue 50
        self.place_in_area(loop_group, "B1", "C6", scale_factor=0.7)
        
        # Distance line AB (re-calculate positions after group placement)
        line_ab = Line(dot_a.get_center(), dot_b.get_center(), color=COLOR_DISTANCE, stroke_width=5)
        
        # Grid setup
        grid_elements = VGroup()
        rows, cols = 6, 6
        for r in range(rows):
            for c in range(cols):
                # Sinusoidal logic to simulate a "mountain range" distance field
                d_val = np.abs(np.sin(r * 0.6) - np.cos(c * 0.6))
                cell_color = interpolate_color(COLOR_SURFACE, ManimColor(BLUE_E), d_val / 2.0)
                
                rect = Square(
                    side_length=0.38, 
                    fill_color=cell_color, 
                    fill_opacity=0, # Initially no fill
                    stroke_width=0.5, 
                    stroke_color=COLOR_SURFACE
                )
                grid_elements.add(rect)
                
        grid_elements.arrange_in_grid(rows=rows, cols=cols, buff=0.05)
        # Position grid according to Issue 51
        self.place_in_area(grid_elements, "E1", "F6", scale_factor=0.8)
        
        # Highlight marker for current pair
        target_cell = grid_elements[1 * cols + 4] # cell at row 1, col 4
        highlight_marker = Dot(target_cell.get_center(), color=COLOR_POINTS, radius=0.06)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_POINTS))
        self.play(Create(loop))
        self.play(FadeIn(dot_a, dot_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_DISTANCE))
        self.play(Create(line_ab))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_SURFACE))
        self.play(FadeIn(grid_elements))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_SURFACE))
        # Animate the appearance of the color/surface data
        self.play(*[rect.animate.set_fill(opacity=0.9) for rect in grid_elements], run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_SURFACE))
        self.play(FadeIn(highlight_marker))
        self.wait(2)
