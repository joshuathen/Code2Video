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
        # Define constants
        BLUE_COLOR = "#3498DB"
        ORANGE_COLOR = "#E67E22"
        WHITE_COLOR = "#ECF0F1"

        lecture_lines = [
            "Independence means one event never affects another.",
            "Robot battery levels are separate and independent events.",
            "Multiply individual chances to find the joint probability."
        ]

        self.setup_layout("The Foundation: Understanding Independence", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(BLUE_COLOR))

        # Create City A and City B with city icons [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/city.svg]
        city_a = Circle(radius=1.2, color=BLUE_COLOR, stroke_width=4)
        city_b = Circle(radius=1.2, color=ORANGE_COLOR, stroke_width=4)
        
        city_icon_a = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/city.svg", color=BLUE_COLOR, fill_opacity=0.2)
        city_icon_b = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/city.svg", color=ORANGE_COLOR, fill_opacity=0.2)

        label_a = Text("City A", font_size=24, color=BLUE_COLOR)
        label_b = Text("City B", font_size=24, color=ORANGE_COLOR)

        # Fix: Shift the circle containers down to start from Row C (Issue 28)
        self.place_in_area(city_a, "C1", "E3")
        self.place_in_area(city_b, "C4", "E6")
        
        self.place_in_area(city_icon_a, "C1", "E3", scale_factor=0.6)
        self.place_in_area(city_icon_b, "C4", "E6", scale_factor=0.6)
        
        # Fix: Ensure labels are centered in Row B with a reduced scale (Issue 29)
        self.place_at_grid(label_a, "B2", scale_factor=0.7)
        self.place_at_grid(label_b, "B5", scale_factor=0.7)

        self.play(
            Create(city_a), 
            Create(city_b),
            FadeIn(city_icon_a),
            FadeIn(city_icon_b),
            Write(label_a),
            Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_COLOR)
        )

        # Represent Bolt and Nut using robot icons [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        robot_bolt = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg", color=WHITE_COLOR)
        robot_nut = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg", color=WHITE_COLOR)
        
        bolt_name = Text("Bolt", font_size=20, color=WHITE_COLOR)
        nut_name = Text("Nut", font_size=20, color=WHITE_COLOR)
        
        prob_bolt = Text("50%", font_size=24, color=WHITE_COLOR)
        prob_nut = Text("50%", font_size=24, color=WHITE_COLOR)

        # Fix: Distribute the name, icon, and probability into separate grid cells (Issue 30)
        self.place_at_grid(bolt_name, "C2", scale_factor=0.8)
        self.place_at_grid(robot_bolt, "D2", scale_factor=0.6)
        self.place_at_grid(prob_bolt, "E2", scale_factor=0.8)
        
        self.place_at_grid(nut_name, "C5", scale_factor=0.8)
        self.place_at_grid(robot_nut, "D5", scale_factor=0.6)
        self.place_at_grid(prob_nut, "E5", scale_factor=0.8)

        self.play(
            FadeIn(robot_bolt), FadeIn(robot_nut),
            Write(bolt_name), Write(nut_name),
            Write(prob_bolt), Write(prob_nut)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE_COLOR)
        )

        # Equation P(A ∩ B) = 0.5 × 0.5 = 0.25
        equation = Text(
            "P(A ∩ B) = 0.5 × 0.5 = 0.25", 
            font_size=36, 
            color=WHITE_COLOR
        )
        # Position at the top center using Row A area (Issue 28)
        self.place_in_area(equation, "A2", "A5")

        self.play(Write(equation))
        self.wait(2)
