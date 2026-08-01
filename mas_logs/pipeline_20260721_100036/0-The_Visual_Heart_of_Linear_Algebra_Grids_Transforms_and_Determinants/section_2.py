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
        title_text = "The DNA of Space: Basis Vectors"
        lecture_lines = [
            "Every vector is built from two fundamental pieces.",
            "Basis vector \"i\" points one unit right.",
            "Basis vector \"j\" points one unit up.",
            "All vectors are combinations of these two units.",
            "Together, they form the DNA of our space."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Calculate visual origin (midpoint of the 6x6 grid area on the right)
        # Midpoint between C3 and D4 in the A1-F6 grid system
        origin = (self.grid["C3"] + self.grid["D4"]) / 2

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 (#00FFCC). Display a standard grid (#555555).
        self.play(self.lecture[0].animate.set_color("#00FFCC"))
        
        grid = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": "#555555",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"include_tip": False, "stroke_color": "#555555"}
        )
        self.place_in_area(grid, "A1", "F6")
        
        self.play(Create(grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 (#00FFCC). Draw a red arrow (#FF0000) from (0,0) to (1,0) and label it "i-hat".
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFCC")
        )
        
        i_hat_vec = Arrow(origin, origin + RIGHT, buff=0, color="#FF0000", stroke_width=4)
        i_hat_label = MathTex(r"\hat{i}", color="#FF0000")
        self.place_at_grid(i_hat_label, "D4", scale_factor=0.8)
        
        self.play(GrowArrow(i_hat_vec))
        self.play(FadeIn(i_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 (#00FFCC). Draw a green arrow (#00FF00) from (0,0) to (0,1) and label it "j-hat".
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFCC")
        )
        
        j_hat_vec = Arrow(origin, origin + UP, buff=0, color="#00FF00", stroke_width=4)
        j_hat_label = MathTex(r"\hat{j}", color="#00FF00")
        # Fix for Issue 19: Use B2 for positioning tip-aligned label
        self.place_at_grid(j_hat_label, "B2", scale_factor=0.8)
        
        self.play(GrowArrow(j_hat_vec))
        self.play(FadeIn(j_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4 (#00FFCC). Show the vector [3, -2] being composed by 3 red i-hat arrows and 2 inverted green j-hat arrows.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FFCC")
        )
        
        red_components = VGroup(*[
            Arrow(origin + k * RIGHT, origin + (k + 1) * RIGHT, buff=0, color="#FF0000", stroke_width=2)
            for k in range(3)
        ])
        
        green_components = VGroup(*[
            Arrow(origin + 3 * RIGHT + k * DOWN, origin + 3 * RIGHT + (k + 1) * DOWN, buff=0, color="#00FF00", stroke_width=2)
            for k in range(2)
        ])
        
        resultant_vec = Arrow(origin, origin + 3 * RIGHT + 2 * DOWN, buff=0, color=YELLOW, stroke_width=5)
        resultant_label = Matrix([[3], [-2]]).set_color(YELLOW)
        # Fix for Issue 20: Use area E5-F6 to avoid edge clipping and improve layout
        self.place_in_area(resultant_label, "E5", "F6", scale_factor=0.7)
        
        self.play(LaggedStart(*[GrowArrow(arr) for arr in red_components], lag_ratio=0.5))
        self.play(LaggedStart(*[GrowArrow(arr) for arr in green_components], lag_ratio=0.5))
        self.play(GrowArrow(resultant_vec))
        self.play(FadeIn(resultant_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5 (#00FFCC). The entire grid pulses in white (#FFFFFF) to emphasize that i and j are the basis.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#00FFCC")
        )
        
        self.play(
            grid.animate.set_stroke(color="#FFFFFF", opacity=1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
