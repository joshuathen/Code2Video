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
        # Setup layout
        self.setup_layout(
            "Prerequisite Tool: The Chain Rule Review",
            [
                "The chain rule handles functions nested inside others.",
                "Differentiating y cubed gives three y squared times dy dx.",
                "Always multiply by the derivative of the inner function."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line in Yellow
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Show chain rule general formula in yellow (Using Text to avoid LaTeX dependency)
        # d/dx [f(x)]^3 = 3[f(x)]^2 * f'(x)
        formula1 = Text(
            "d/dx [f(x)]^3 = 3[f(x)]^2 * f'(x)",
            font_size=24,
            color="#FFFF00"
        )
        self.place_in_area(formula1, "A1", "B6", scale_factor=0.9)
        self.play(Write(formula1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line in Red
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        # Transformation: d/dx[y^3] -> 3y^2 * (dy/dx)
        expr_start = Text("d/dx [y^3]", font_size=28)
        # Using t2c for internal coloring in Text for dy/dx
        expr_end = Text("3y^2 * dy/dx", font_size=28, t2c={"dy/dx": "#FF0000"})
        
        self.place_in_area(expr_start, "C1", "D6", scale_factor=1.0)
        self.place_in_area(expr_end, "C1", "D6", scale_factor=1.0)
        
        self.play(Write(expr_start))
        self.wait(1)
        self.play(ReplacementTransform(expr_start, expr_end))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line in Green
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        def create_gear(color, label_text):
            # Base gear shape
            core = Circle(radius=0.35, color=color, fill_opacity=0.3)
            # Add teeth for gear visual effect
            num_teeth = 8
            teeth = VGroup()
            for i in range(num_teeth):
                tooth = Rectangle(width=0.15, height=0.15, color=color, fill_opacity=0.8)
                angle = i * (TAU / num_teeth)
                # Move to edge of circle
                pos = core.point_at_angle(angle)
                tooth.move_to(pos)
                tooth.rotate(angle)
                teeth.add(tooth)
            
            # Label inside the gear using Text
            label = Text(label_text, font_size=18, color=WHITE).move_to(core.get_center())
            return VGroup(core, teeth, label)

        # Three gears representing the chain rule relationship
        gear_x = create_gear("#0000FF", "x")        # Blue
        gear_y = create_gear("#00FF00", "y")        # Green (Matches Line 3 color)
        gear_dy = create_gear("#FF0000", "dy/dx")   # Red

        # Place gears horizontally on the grid
        self.place_at_grid(gear_x, "E2", scale_factor=1.0)
        self.place_at_grid(gear_y, "E3", scale_factor=1.0)
        self.place_at_grid(gear_dy, "E4", scale_factor=1.0)

        self.play(
            FadeIn(gear_x),
            FadeIn(gear_y),
            FadeIn(gear_dy)
        )
        
        # Rotating gears together to show dependency
        # Middle gear (y) rotates in opposite direction to mesh
        self.play(
            Rotate(gear_x, angle=2*PI),
            Rotate(gear_y, angle=-2*PI),
            Rotate(gear_dy, angle=2*PI),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
