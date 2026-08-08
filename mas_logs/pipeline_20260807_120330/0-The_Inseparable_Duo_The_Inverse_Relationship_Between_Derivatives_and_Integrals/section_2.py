from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Data from storyboard
        title = "Prerequisite Recap: Slope and Area"
        lecture_lines = [
            "- Recall that derivatives represent the slope of curves.",
            "- In contrast, integrals represent the area under curves.",
            "- We will now link these two fundamental concepts."
        ]
        
        # Initialize layout
        self.setup_layout(title, lecture_lines)
        
        # Colors from storyboard
        GREEN_CURVE = "#00FF00"
        MAGENTA_SLOPE = "#FF00FF"
        LIGHT_GREEN_AREA = "#90EE90"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Magenta (matching tangent/slope visual)
        self.play(self.lecture[0].animate.set_color(MAGENTA_SLOPE))

        # Setup Axes in the right-side grid area (A1 to F6)
        # Issue 31: scale_factor=0.85
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE_COLOR}
        ).set_z_index(0)
        self.place_in_area(axes, "A1", "F6", scale_factor=0.85)
        
        # Parabolic curve: y = 0.2x^2 - 1
        curve = axes.plot(lambda x: 0.2 * x**2 - 1, color=GREEN_CURVE)
        
        # ValueTracker for the tangent point x-coordinate
        t_val = ValueTracker(-2)
        
        # Dot at the tangent point
        dot = Dot(color=MAGENTA_SLOPE).set_z_index(2)
        dot.add_updater(lambda m: m.move_to(axes.c2p(t_val.get_value(), 0.2 * t_val.get_value()**2 - 1)))
        
        # Tangent line updated dynamically
        tangent_line = Line(color=MAGENTA_SLOPE).set_z_index(1)
        def update_tangent(m):
            x = t_val.get_value()
            y = 0.2 * x**2 - 1
            slope = 0.4 * x
            # Determine start and end points for a fixed length tangent
            p1 = axes.c2p(x - 0.8, y - slope * 0.8)
            p2 = axes.c2p(x + 0.8, y + slope * 0.8)
            m.put_start_and_end_on(p1, p2)
        tangent_line.add_updater(update_tangent)

        self.play(Create(axes), Create(curve))
        self.add(dot, tangent_line)
        # Animate the tangent sliding along the curve
        self.play(t_val.animate.set_value(0.5), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in Light Green (matching area visual)
        self.play(self.lecture[1].animate.set_color(LIGHT_GREEN_AREA))
        
        # Create and animate the shaded area under the curve
        # Area from x = 0.5 to x = 2.5
        shaded_area = axes.get_area(curve, x_range=[0.5, 2.5], color=LIGHT_GREEN_AREA, opacity=0.4)
        
        self.play(FadeIn(shaded_area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to linking the concepts
        self.play(self.lecture[2].animate.set_color(WHITE_COLOR))
        
        # Create labels as per storyboard: "Slope (Derivative)" and "Area (Integral)"
        label_slope = Text("Slope (Derivative)", font_size=24, color=WHITE_COLOR)
        label_area = Text("Area (Integral)", font_size=24, color=WHITE_COLOR)
        
        # Position labels using grid
        # Issue 29: place_at_grid(label_slope, 'A2', scale_factor=0.7)
        # Issue 30: place_at_grid(label_area, 'F5', scale_factor=0.7)
        self.place_at_grid(label_slope, "A2", scale_factor=0.7)
        self.place_at_grid(label_area, "F5", scale_factor=0.7)
        
        # Create arrows pointing to the visuals
        arrow_slope = Arrow(
            start=label_slope.get_bottom(),
            end=dot.get_center(),
            buff=0.1,
            color=WHITE_COLOR,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )
        
        # Point to the center of the shaded area
        area_center = axes.c2p(1.5, -0.5) # Approximate center of shaded region
        arrow_area = Arrow(
            start=label_area.get_top(),
            end=area_center,
            buff=0.1,
            color=WHITE_COLOR,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )

        self.play(
            Write(label_slope),
            Write(label_area),
            Create(arrow_slope),
            Create(arrow_area)
        )
        self.wait(2)
