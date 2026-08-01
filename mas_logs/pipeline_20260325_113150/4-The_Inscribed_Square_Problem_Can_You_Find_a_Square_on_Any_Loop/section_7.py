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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize the layout with title and lecture lines
        lecture_lines = [
            "Rectangles exist for all smooth Jordan curves.",
            "For fractal-like loops, the square version remains open.",
            "Mathematics still has many beautiful mysteries to solve."
        ]
        self.setup_layout("Summary and Real-World Status", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Cyan
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=1)
        
        # Create a smooth Jordan curve (Ellipse)
        ellipse = Ellipse(width=3.5, height=2.0, color=WHITE)
        # Create a cyan rectangle inscribed in the ellipse
        # Vertex formula for ellipse (a cos t, b sin t)
        a_val, b_val = 1.75, 1.0
        t_param = PI / 4
        rect_points = [
            [a_val * np.cos(t_param), b_val * np.sin(t_param), 0],
            [-a_val * np.cos(t_param), b_val * np.sin(t_param), 0],
            [-a_val * np.cos(t_param), -b_val * np.sin(t_param), 0],
            [a_val * np.cos(t_param), -b_val * np.sin(t_param), 0]
        ]
        cyan_rect = Polygon(*rect_points, color="#00FFFF")
        
        # Group and position in the top-right area of the grid
        smooth_group = VGroup(ellipse, cyan_rect)
        self.place_in_area(smooth_group, "A2", "C5", scale_factor=0.8)
        
        self.play(Create(ellipse))
        self.play(Create(cyan_rect))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in Yellow
        self.play(self.lecture[1].animate.set_color("#FFFF00"), run_time=1)
        
        # Create a fractal-like jagged loop using deterministic "random" points
        np.random.seed(123)
        n_pts = 50
        fractal_points = []
        for i in range(n_pts):
            angle = i * (TAU / n_pts)
            # Add some jaggedness to a circle
            radius = 1.2 + 0.3 * np.random.rand()
            fractal_points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        fractal_loop = VMobject(color=WHITE)
        fractal_loop.set_points_as_corners([*fractal_points, fractal_points[0]])
        
        # Create a yellow question mark representing the open problem
        question_mark = Text("?", color="#FFFF00", font_size=72)
        
        # Group and position in the bottom-right area of the grid
        fractal_group = VGroup(fractal_loop, question_mark)
        self.place_in_area(fractal_group, "D2", "F5", scale_factor=0.8)
        
        self.play(Create(fractal_loop))
        self.play(Write(question_mark))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Ensure lecture line 3 is highlighted (White)
        self.play(self.lecture[2].animate.set_color("#FFFFFF"), run_time=1)
        
        # Names of significant mathematicians in this field
        name_vaughan = Text("Vaughan", color="#FFFFFF", font_size=36)
        name_stromquist = Text("Stromquist", color="#FFFFFF", font_size=36)
        names_vgroup = VGroup(name_vaughan, name_stromquist).arrange(DOWN, buff=0.5)
        
        # Position names centrally within the animation area
        self.place_in_area(names_vgroup, "B2", "E5", scale_factor=1.0)
        
        # Clear previous curves and show names
        self.play(
            FadeOut(smooth_group),
            FadeOut(fractal_group),
            Write(names_vgroup)
        )
        self.wait(2)
        
        # Final fade to black for the conclusion of the video
        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
        self.wait(1)
