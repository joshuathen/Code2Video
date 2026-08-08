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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initializing titles and lecture lines
        title = "Prerequisite Recap: The Two Pillars"
        lines = [
            "Calculus rests on two fundamental pillars.",
            "Derivatives measure the instantaneous rate of change.",
            "Integrals calculate the total accumulated area."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Line 1: Introduction to the curve and tangent line.
        # Storyboard: Display curve f(x) = x^2 with a red (#FF0000) tangent line at x=2.
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Position axes in the visual area (B2 to E6)
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        
        graph = axes.plot(lambda x: x**2, x_range=[0, 2.2], color=BLUE)
        
        # Tangent line at x=2 (slope = 4, point = (2,4))
        tangent_line = Line(
            axes.c2p(1.7, 4 * 1.7 - 4),
            axes.c2p(2.2, 4 * 2.2 - 4),
            color="#FF0000",
            stroke_width=5
        )
        
        self.play(Create(axes), Create(graph))
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Focus on derivatives (instantaneous rate of change).
        # Storyboard: Shade the area under the curve in green (#00FF00).
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF0000") # Matching derivative color
        )
        
        # Adding the integral visualization (Area under the curve)
        area = axes.get_area(graph, x_range=[0, 2], color="#00FF00", opacity=0.4)
        
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Focus on integrals (total accumulated area).
        # Storyboard: Show a Cheetah icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg] 
        # with a speedometer (derivative) and distance counter (integral).
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00") # Matching integral color
        )

        # Cheetah SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg]
        cheetah_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_in_area(cheetah_svg, "E1", "E2", scale_factor=0.6)

        # Label for the Cheetah (Issue 31: Place in area F1-F2)
        cheetah_label = Text("Dash the Cheetah", font_size=20, color=YELLOW)
        self.place_in_area(cheetah_label, "F1", "F2", scale_factor=0.8)
        
        # Speedometer representing Derivative (Issue 30: Place in area F3-F4)
        speedometer = MathTex(r"v(t) = \frac{dx}{dt}", font_size=22, color="#FF0000")
        speed_text = Text("Speedometer", font_size=16, color="#FF0000").next_to(speedometer, UP, buff=0.1)
        speed_group = VGroup(speedometer, speed_text)
        self.place_in_area(speed_group, "F3", "F4", scale_factor=0.8)
        
        # Distance counter representing Integral (Issue 29: Place in area F5-F6)
        distance = MathTex(r"s = \int v(t) dt", font_size=22, color="#00FF00")
        dist_text = Text("Odometer", font_size=16, color="#00FF00").next_to(distance, UP, buff=0.1)
        dist_group = VGroup(distance, dist_text)
        self.place_in_area(dist_group, "F5", "F6", scale_factor=0.8)

        self.play(
            FadeIn(cheetah_svg),
            FadeIn(cheetah_label),
            FadeIn(speed_group),
            FadeIn(dist_group)
        )
        self.wait(3)
