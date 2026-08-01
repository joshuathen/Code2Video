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
        # Setup layout with title and lecture lines
        self.setup_layout("Prerequisite: The Math of a Single Decision", [
            "A single decision starts with a simple linear equation.",
            "Weights adjust importance, while bias sets the threshold.",
            "This mathematical line separates data into distinct categories."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Equation: y = wx + b
        # Using explicit pieces for easier coloring/animation later
        equation = MathTex("y", "=", "w", "x", "+", "b", font_size=48, color=WHITE)
        self.place_in_area(equation, "B2", "B5")
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        # Line 2 corresponds to weights and bias colors
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Labels for w (Weight) and b (Bias)
        w_part = equation[2]
        b_part = equation[5]
        
        w_label = Text("Weight/Importance", font_size=20, color="#FFD700")
        b_label = Text("Bias/Threshold", font_size=20, color="#FF69B4")
        
        # FIX Issue 35: Use area positioning for multi-word labels
        self.place_in_area(w_label, 'A1', 'A3', scale_factor=0.7)
        self.place_in_area(b_label, 'A4', 'A6', scale_factor=0.7)
        
        w_arrow = Arrow(w_label.get_bottom(), w_part.get_top(), color="#FFD700", buff=0.1)
        b_arrow = Arrow(b_label.get_bottom(), b_part.get_top(), color="#FF69B4", buff=0.1)
        
        # Input nodes visualization (Weather and Distance)
        weather_node = Circle(radius=0.4, color=WHITE)
        weather_text = Text("Weather", font_size=18).next_to(weather_node, UP, buff=0.1)
        weather_group = VGroup(weather_node, weather_text)
        
        dist_node = Circle(radius=0.4, color=WHITE)
        dist_text = Text("Distance", font_size=18).next_to(dist_node, UP, buff=0.1)
        dist_group = VGroup(dist_node, dist_text)
        
        # FIX Issue 36: Use area positioning for asset groups
        self.place_in_area(weather_group, 'D1', 'D3', scale_factor=0.9)
        self.place_in_area(dist_group, 'E1', 'E3', scale_factor=0.9)
        
        # Load decimal numbers once (avoid creation in updaters)
        weather_val = DecimalNumber(0.8, num_decimal_places=1, color=YELLOW).scale(0.7)
        dist_val = DecimalNumber(0.4, num_decimal_places=1, color=YELLOW).scale(0.7)
        weather_val.move_to(weather_node.get_center())
        dist_val.move_to(dist_node.get_center())

        self.play(
            Create(w_arrow), FadeIn(w_label),
            Create(b_arrow), FadeIn(b_label),
            w_part.animate.set_color("#FFD700"),
            b_part.animate.set_color("#FF69B4")
        )
        self.play(
            FadeIn(weather_group), 
            FadeIn(dist_group), 
            FadeIn(weather_val), 
            FadeIn(dist_val)
        )
        
        # Dynamic value change to illustrate importance/weights
        self.play(
            ChangeDecimalToValue(weather_val, 0.9),
            ChangeDecimalToValue(dist_val, 0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line and clear middle-stage visuals
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#00FFFF"),
            FadeOut(w_arrow), FadeOut(w_label),
            FadeOut(b_arrow), FadeOut(b_label),
            FadeOut(weather_group), FadeOut(dist_group),
            FadeOut(weather_val), FadeOut(dist_val),
            equation.animate.scale(0.6).move_to(self.grid["A6"])
        )
        
        # Coordinate plane setup
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, "C2", "F5")
        
        # Area classification labels
        stay_home = Text("Stay Home", font_size=20, color=RED)
        go_park = Text("Go to Park", font_size=20, color=GREEN)
        
        # FIX Issue 37: Avoid overlap with axis by moving stay_home label
        self.place_at_grid(stay_home, 'E3', scale_factor=0.7) 
        self.place_at_grid(go_park, "D5", scale_factor=0.8)  
        
        # Linear separator: y = x + 1 (conceptual)
        line = axes.plot(lambda x: x + 1, color="#00FFFF", x_range=[0, 4])
        
        self.play(Create(axes))
        self.play(Create(line))
        self.play(FadeIn(stay_home), FadeIn(go_park))
        
        # Decision point movement
        dot = Dot(axes.c2p(1, 0.5), color=WHITE)
        dot_label = Text("Decision", font_size=16)
        # Use updater to keep label next to moving dot
        dot_label.add_updater(lambda m: m.next_to(dot, RIGHT, buff=0.1))
        
        self.play(FadeIn(dot), FadeIn(dot_label))
        
        # Move the point across the decision boundary
        self.play(
            dot.animate.move_to(axes.c2p(1, 3.5)),
            run_time=2
        )
        self.play(
            dot.animate.move_to(axes.c2p(3, 4.5)),
            run_time=2
        )
        
        # Region highlighting
        area = axes.get_area(line, x_range=[0, 4], color=GREEN, opacity=0.2)
        self.play(FadeIn(area))
        
        self.wait(2)
