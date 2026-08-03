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
        # 1. Setup Layout and Titles
        self.setup_layout(
            "Example Walkthrough: The Leaky Tank",
            [
                "Consider a tank leaking water at a changing rate.",
                "Integrating the flow rate gives the total volume lost.",
                "Calculus provides exact answers for these real-world problems."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color Line 1
        self.play(self.lecture[0].animate.set_color(BLUE), run_time=1)

        # Load Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg]
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg", color=GREY_A)
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg", color=BLUE_B)
        
        self.place_in_area(tank, "B1", "D3", scale_factor=1.5)
        self.place_in_area(water, "B1", "D3", scale_factor=1.5)

        # ValueTracker for water level (0 to 1)
        water_level = ValueTracker(1.0)
        
        # Capture initial water properties for scaling
        initial_water_height = water.height
        bottom_y = water.get_bottom()[1]

        def update_water(m):
            new_height = initial_water_height * water_level.get_value()
            if new_height > 0.01:
                m.stretch_to_fit_height(new_height)
                # Reposition bottom back to original bottom
                m.move_to([m.get_x(), bottom_y + new_height/2, 0])
            else:
                m.set_opacity(0)

        water.add_updater(update_water)

        self.play(FadeIn(tank), FadeIn(water), run_time=1.5)
        self.play(water_level.animate.set_value(0.3), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF4500"),
            run_time=1
        )

        # Plot rate function v(t) = 2t
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 20, 5],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": GREY_C},
            tips=False
        ).add_coordinates()
        self.place_in_area(axes, "C4", "F6", scale_factor=1.0)

        # v(t) = 2t plot
        v_graph = axes.plot(lambda t: 2*t, x_range=[0, 10], color="#FF4500")
        v_label = MathTex("v(t) = 2t", color="#FF4500")
        # Issue 46 Fix: func_label at A4 with scale 0.7
        self.place_at_grid(v_label, "A4", scale_factor=0.7)

        # Integral area
        area = axes.get_area(v_graph, x_range=[0, 10], color=BLUE_E, opacity=0.5)
        area_label = MathTex(r"V = \int_0^{10} 2t \, dt", color=BLUE_B)
        # Issue 45 Fix: area_label at A5 with scale 0.7
        self.place_at_grid(area_label, "A5", scale_factor=0.7)

        self.play(Create(axes), run_time=1)
        self.play(Create(v_graph), Write(v_label), run_time=1)
        self.play(FadeIn(area), Write(area_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            run_time=1
        )

        # Result calculation
        result_tex = MathTex(r"= [t^2]_0^{10} = 100 \text{ L}", color=YELLOW)
        # Issue 44 Fix: result_tex at A6 with scale 0.7
        self.place_at_grid(result_tex, "A6", scale_factor=0.7)

        self.play(Write(result_tex), run_time=2)
        self.wait(3)

        # Cleanup
        water.remove_updater(update_water)
