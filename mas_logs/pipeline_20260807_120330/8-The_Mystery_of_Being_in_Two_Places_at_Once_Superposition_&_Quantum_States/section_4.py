from manim import *
import numpy as np

# Base class provided in instructions
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
        title_text = "The Role of Measurement (The Collapse)"
        lecture_lines = [
            "We can never directly observe a superposition.",
            "Measuring a quantum state forces a definite outcome.",
            "This process is called wavefunction collapse.",
            "The vector snaps instantly to |0⟩ or |1⟩.",
            "Probability dictates which result we finally see."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color palette for lecture lines and matching elements
        COLOR_SUPERPOSITION = "#88C0D0" # Light blue
        COLOR_MEASUREMENT = "#CCCCCC"   # Light grey
        COLOR_COLLAPSE = "#A3BE8C"      # Sage green
        COLOR_OUTCOME = "#EBCB8B"       # Muted yellow
        COLOR_PROBABILITY = "#FFA500"   # Orange

        # === Animation for Lecture Line 1 ===
        # Show the diagonal vector from Section 2 in its unit circle.
        self.lecture[0].set_color(COLOR_SUPERPOSITION)
        
        # Grid center for unit circle area (B4 to D6) - Moved per Issue 32 to avoid overlapping lecture text
        circle_area_center = self.place_in_area(Dot(radius=0, fill_opacity=0), "B4", "D6").get_center()
        
        unit_circle = Circle(radius=1.2, color=GREY_A).move_to(circle_area_center)
        axes = Axes(
            x_range=[-1.2, 1.2, 1],
            y_range=[-1.2, 1.2, 1],
            x_length=2.4,
            y_length=2.4,
            axis_config={"include_tip": True, "color": GREY_B, "stroke_width": 2}
        ).move_to(circle_area_center)
        
        # Axis labels (using B001 to avoid overlap)
        label_0 = MathTex(r"|0\rangle", color=COLOR_OUTCOME, font_size=24)
        label_1 = MathTex(r"|1\rangle", color=COLOR_OUTCOME, font_size=24)
        label_0.next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        label_1.next_to(axes.y_axis.get_end(), UP, buff=0.1)
        
        # Diagonal vector (45 degrees)
        state_vector = Arrow(
            start=circle_area_center,
            end=axes.c2p(np.cos(PI/4), np.sin(PI/4)),
            buff=0,
            color=COLOR_SUPERPOSITION,
            stroke_width=4
        )
        
        self.play(Create(axes), Create(unit_circle), FadeIn(label_0, label_1))
        self.play(GrowArrow(state_vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A camera icon (color: #CCCCCC) appears labeled 'Measurement'.
        self.lecture[1].set_color(COLOR_MEASUREMENT)
        
        # Camera Icon Construction
        cam_body = RoundedRectangle(corner_radius=0.05, height=0.4, width=0.6, color=COLOR_MEASUREMENT, fill_opacity=1)
        cam_lens = Circle(radius=0.1, color=BLACK, fill_opacity=1).move_to(cam_body.get_center())
        cam_button = Rectangle(height=0.05, width=0.15, color=COLOR_MEASUREMENT, fill_opacity=1).next_to(cam_body, UP, buff=0, aligned_edge=RIGHT)
        cam_icon = VGroup(cam_body, cam_lens, cam_button)
        cam_text = Text("Measurement", font_size=14, color=COLOR_MEASUREMENT).next_to(cam_icon, DOWN, buff=0.1)
        measurement_tool = VGroup(cam_icon, cam_text)
        
        # Position changed to A5 per Issue 33 to distinguish from the vector group
        self.place_at_grid(measurement_tool, "A5")
        
        self.play(FadeIn(measurement_tool))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This process is called wavefunction collapse.
        self.lecture[2].set_color(COLOR_COLLAPSE)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The camera shutter clicks (flash effect).
        # The diagonal vector 'snaps' instantly to the X-axis (|0⟩).
        self.lecture[3].set_color(COLOR_OUTCOME)
        
        flash = Rectangle(width=14, height=8, fill_color=WHITE, fill_opacity=0.7, stroke_width=0)
        
        # Target state is |0>
        snapped_vector = Arrow(
            start=circle_area_center,
            end=axes.c2p(1, 0),
            buff=0,
            color=COLOR_OUTCOME,
            stroke_width=4
        )
        
        # Shutter click logic
        self.play(FadeIn(flash, run_time=0.1))
        # Snap vector while flash is brightest
        state_vector.set_color(COLOR_OUTCOME)
        self.play(Transform(state_vector, snapped_vector), run_time=0.1)
        self.play(FadeOut(flash, run_time=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A bar chart (color: #FFA500) shows probability for |0⟩ at 100% and |1⟩ at 0%.
        self.lecture[4].set_color(COLOR_PROBABILITY)
        
        # Bar chart components
        prob_axes = Axes(
            x_range=[0, 2, 1],
            y_range=[0, 1.2, 0.5],
            x_length=2.0,
            y_length=1.5,
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        
        chart_label_0 = MathTex(r"|0\rangle", font_size=18).next_to(prob_axes.c2p(0.5, 0), DOWN, buff=0.1)
        chart_label_1 = MathTex(r"|1\rangle", font_size=18).next_to(prob_axes.c2p(1.5, 0), DOWN, buff=0.1)
        
        # Bars
        # Height of 1.0 on axis
        h_max = prob_axes.c2p(0, 1)[1] - prob_axes.c2p(0, 0)[1]
        
        bar_0 = Rectangle(width=0.4, height=h_max, fill_color=COLOR_PROBABILITY, fill_opacity=1, stroke_width=1)
        bar_0.move_to(prob_axes.c2p(0.5, 0), aligned_edge=DOWN)
        
        bar_1 = Rectangle(width=0.4, height=0.01, fill_color=COLOR_PROBABILITY, fill_opacity=1, stroke_width=1)
        bar_1.move_to(prob_axes.c2p(1.5, 0), aligned_edge=DOWN)
        
        chart_group = VGroup(prob_axes, chart_label_0, chart_label_1, bar_0, bar_1)
        
        # Position changed to E5-F6 per Issue 34 to avoid overlap with vector visual
        self.place_in_area(chart_group, "E5", "F6", scale_factor=1.0)
        
        self.play(FadeIn(prob_axes, chart_label_0, chart_label_1))
        self.play(GrowFromEdge(bar_0, DOWN), GrowFromEdge(bar_1, DOWN))
        self.wait(2)
