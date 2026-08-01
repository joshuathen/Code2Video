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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Summary: Complexity from Simplicity",
            [
                "A simple rule generates infinite fractal beauty.",
                "Small patterns mirror the structure of the whole.",
                "Complex dynamics reveal the order within chaos."
            ]
        )

        # Colors (L008)
        FRACTAL_COLOR = "#00FF00"
        EQUATION_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Grow a green fractal tree (#00FF00) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg] 
        # starting from a single trunk, branching out recursively.
        
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Load and place tree asset (Issue 20)
        # Using self.tree_group as per Issue 34 suggestion
        self.tree_group = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg")
        self.tree_group.set_color(FRACTAL_COLOR)
        
        # Fix Issue 34: Use area B3-F6 and scale 0.7 to avoid overlap with lecture
        self.place_in_area(self.tree_group, 'B3', 'F6', scale_factor=0.7)
        
        # Animate "growing" - Create on SVGMobject follows paths
        self.play(Create(self.tree_group, run_time=3))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Perform a 4-second 'smooth' zoom into a branch of the tree to show it looks identical to the whole tree.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Pick a point on a "branch" to zoom into (approximating UR quadrant of the SVG)
        zoom_point = self.tree_group.get_corner(UR) * 0.4 + self.tree_group.get_center() * 0.6
        
        # L024: Use rate_functions prefix for slow_into
        self.play(
            self.tree_group.animate(run_time=4, rate_func=rate_functions.slow_into).scale(4, about_point=zoom_point)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Display the equation 'z = z^2 + c' (#FFFFFF) and animate it scaling up and down slowly.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # MathTex with fallback (L022)
        try:
            self.equation_simple = MathTex("z = z^2 + c", color=EQUATION_COLOR)
        except Exception:
            self.equation_simple = Text("z = z^2 + c", color=EQUATION_COLOR)
            
        # Fix Issue 35: Use grid A4 and scale 1.0 to avoid overlap with branches
        self.place_at_grid(self.equation_simple, 'A4', scale_factor=1.0)
        
        self.play(FadeIn(self.equation_simple))
        
        # Pulse animation (L024: use rate_functions.there_and_back)
        self.play(
            self.equation_simple.animate(run_time=3, rate_func=rate_functions.there_and_back).scale(1.2)
        )
        self.wait(2.0)
        
        # Final cleanup for smooth ending
        self.play(
            FadeOut(self.equation_simple),
            FadeOut(self.tree_group),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(2.0)
