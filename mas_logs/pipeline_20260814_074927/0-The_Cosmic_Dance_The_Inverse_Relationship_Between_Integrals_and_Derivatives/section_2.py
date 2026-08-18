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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("The Derivative: Zooming into the Slope", [
            "- Let's zoom into a single point on the curve.",
            "- Up close, every smooth curve looks like a line.",
            "- This line's slope is the derivative, or instantaneous change."
        ])
        
        # Colors
        blue_curve_color = "#00BFFF"
        white_color = "#FFFFFF"
        orange_color = "#FF4500"

        # === Animation for Lecture Line 1 ===
        # Highlight first line in blue
        self.play(self.lecture[0].animate.set_color(blue_curve_color))

        # Setup Axes and Curve in the right-side grid area
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-0.5, 3],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "color": GRAY_D}
        )
        self.place_in_area(axes, "B1", "F6")
        
        def func(x):
            return 0.5 * (x**2) + 0.5
            
        curve = axes.plot(func, x_range=[-1.8, 1.8], color=blue_curve_color)
        
        # Point to zoom in on (x=1.0, y=1.0)
        zoom_point_val = 1.0
        zoom_point_coords = axes.c2p(zoom_point_val, func(zoom_point_val))
        
        # Magnifying glass circle
        magnifier = Circle(radius=0.5, color=white_color, stroke_width=4)
        magnifier.move_to(zoom_point_coords)

        self.play(Create(axes), Create(curve))
        self.play(Create(magnifier))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to white for the zoom focus
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(white_color)
        )

        # Zoom into the point by scaling axes and curve about the zoom point
        # This makes the curve segment inside the magnifier appear flatter
        zoom_factor = 10
        self.play(
            axes.animate.scale(zoom_factor, about_point=zoom_point_coords),
            curve.animate.scale(zoom_factor, about_point=zoom_point_coords),
            run_time=4
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to orange for the derivative concept
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(orange_color)
        )

        # Draw tangent line. Since we zoomed in at x=1 for f(x)=0.5x^2+0.5, 
        # the slope is f'(1)=1 (a 45-degree angle).
        tangent_line = Line(
            start=zoom_point_coords + np.array([-1.5, -1.5, 0]),
            end=zoom_point_coords + np.array([1.5, 1.5, 0]),
            color=orange_color,
            stroke_width=6
        )
        
        # Label next to the tangent line within the grid area
        label = MathTex(r"f'(x) = \text{Slope}", color=orange_color)
        self.place_at_grid(label, "C5", scale_factor=0.8)
        # Refine position relative to the point of interest
        label.next_to(magnifier, UP + RIGHT, buff=0.1)

        self.play(Create(tangent_line))
        self.play(Write(label))
        self.wait(3)

        # Final cleanup for the section transition
        self.play(
            FadeOut(axes), 
            FadeOut(curve), 
            FadeOut(magnifier), 
            FadeOut(tangent_line), 
            FadeOut(label)
        )
