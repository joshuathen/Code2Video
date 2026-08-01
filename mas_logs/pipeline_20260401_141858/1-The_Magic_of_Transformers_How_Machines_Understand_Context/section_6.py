from manim import *
import numpy as np
import os

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
        # Title and Lecture lines setup
        title_str = "Emergence: From Math to Intelligence"
        lines_str = [
            "Massive scale leads to surprising emergent properties.",
            "Models transition from word-fillers to versatile AI assistants.",
            "Complex reasoning emerges from simple mathematical foundations."
        ]
        self.setup_layout(title_str, lines_str)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)

        # Create a small yellow spark
        spark = Star(n=8, outer_radius=0.3, inner_radius=0.1, color="#FFFF00", fill_opacity=1)
        self.place_at_grid(spark, "C3")
        
        spark_label = Text("Simple Word-Filler", font_size=20, color="#FFFFFF")
        self.place_at_grid(spark_label, "D3")

        self.play(
            Create(spark),
            Write(spark_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFA500"),
            run_time=0.5
        )

        # Expand the spark into a massive sun-like sphere
        sun_sphere = Circle(radius=1.5, color="#FFA500", fill_opacity=0.6)
        # Add a glow effect using multiple circles
        glow = VGroup(*[
            Circle(radius=1.5 + i*0.1, color="#FFA500", stroke_width=0, fill_opacity=0.1)
            for i in range(5)
        ])
        sun_group = VGroup(glow, sun_sphere)
        self.place_in_area(sun_group, "B2", "E5")

        ai_label = Text("AI Assistant", font_size=24, color="#FFFFFF")
        # Fixed grid position: "F3.5" is invalid, using "F3"
        self.place_at_grid(ai_label, "F3")

        self.play(
            ReplacementTransform(spark, sun_group),
            ReplacementTransform(spark_label, ai_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )

        # Animate labels 'Code', 'Poetry', and 'Logic' emerging from the sphere
        code_label = Text("Code", font_size=24, color="#FFFFFF")
        poetry_label = Text("Poetry", font_size=24, color="#FFFFFF")
        logic_label = Text("Logic", font_size=24, color="#FFFFFF")

        # Initial positions inside the sun
        sun_center = sun_group.get_center()
        code_label.move_to(sun_center).scale(0.1)
        poetry_label.move_to(sun_center).scale(0.1)
        logic_label.move_to(sun_center).scale(0.1)

        # Targets - Fixed overlapping issues by moving labels further from the center
        target_code = code_label.copy()
        self.place_at_grid(target_code, "A2", scale_factor=1.0)
        
        target_poetry = poetry_label.copy()
        self.place_at_grid(target_poetry, "A5", scale_factor=1.0)
        
        target_logic = logic_label.copy()
        self.place_at_grid(target_logic, "F5", scale_factor=1.0)

        self.play(
            Transform(code_label, target_code),
            Transform(poetry_label, target_poetry),
            Transform(logic_label, target_logic),
            run_time=2,
            rate_func=smooth
        )
        self.wait(2)
