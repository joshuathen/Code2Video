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
        # Initial layout setup
        title_text = "Application: Transforming 'Yoga Cat'"
        lecture_lines = [
            "Let's apply a shear matrix to our Yoga Cat.",
            "This specific matrix shifts the cat's top horizontally.",
            "Every point on the cat follows the new grid.",
            "A vertex at one, two slides over to three, two.",
            "Matrix math calculates exactly where every point should land."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Coordinate Helper relative to grid point E2
        # E2 is roughly x=1.5, y=-1.8 in scene units
        origin = self.grid['E2']
        
        def get_pos(x, y):
            return origin + x * RIGHT + y * UP

        def shear_pos(x, y):
            # Shear matrix [[1, 1], [0, 1]] -> x' = x + y, y' = y
            return origin + (x + y) * RIGHT + y * UP

        # Define Cat Vertices (Local relative to bottom-left)
        # vertex index 3 is the ear at (1, 2)
        cat_local_pts = [
            [0, 0, 0], [1.2, 0, 0], [1.2, 1, 0], [1, 2, 0], 
            [0.7, 1.4, 0], [0.3, 2, 0], [0, 1, 0], [0, 0, 0]
        ]
        
        cat = VMobject(color="#FF00FF") # Magenta
        cat.set_points_as_corners([get_pos(p[0], p[1]) for p in cat_local_pts])
        
        sheared_cat_pts = [shear_pos(p[0], p[1]) for p in cat_local_pts]

        # === Animation for Lecture Line 1 ===
        # Create a magenta cat-like shape
        self.lecture[0].set_color("#FF00FF")
        self.play(Create(cat), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply shear transformation [[1,1],[0,1]]
        self.lecture[1].set_color("#FF00FF")
        self.play(cat.animate.set_points_as_corners(sheared_cat_pts), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Briefly show sheared grid lines in dark blue (#00008B)
        self.lecture[2].set_color("#00008B")
        
        grid_lines = VGroup()
        # Horizontal lines (stay horizontal under shear)
        for y in [0, 1, 2]:
            grid_lines.add(Line(shear_pos(0, y), shear_pos(3.5, y), color="#00008B", stroke_width=1))
        # Slanted vertical lines (transformed under shear)
        for x in [0, 1, 2, 3]:
            grid_lines.add(Line(shear_pos(x, 0), shear_pos(x, 2), color="#00008B", stroke_width=1))
            
        self.play(FadeIn(grid_lines))
        self.wait(1)
        self.play(FadeOut(grid_lines))

        # === Animation for Lecture Line 4 ===
        # Highlight vertex (1,2) on the cat's ear moving to (3,2)
        self.lecture[3].set_color("#FFFF00") # Yellow vertex highlight
        
        v_start = get_pos(1, 2)
        v_end = shear_pos(1, 2) # (3, 2)
        
        # Path trace
        path_line = Line(v_start, v_end, color="#FFFF00", stroke_width=2).set_stroke(opacity=0.5)
        dot = Dot(v_start, color="#FFFF00", radius=0.08)
        
        self.play(Create(path_line), run_time=0.5)
        self.play(dot.animate.move_to(v_end), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display the calculation next to the cat's ear vertex
        self.lecture[4].set_color(WHITE)
        
        calc_text = Text("(1,2) becomes (3,2)", font_size=20, color=WHITE)
        # Position slightly above the final vertex (B5 is grid cell above C5)
        self.place_at_grid(calc_text, 'B5', scale_factor=0.9)
        
        self.play(Write(calc_text))
        self.wait(2)
