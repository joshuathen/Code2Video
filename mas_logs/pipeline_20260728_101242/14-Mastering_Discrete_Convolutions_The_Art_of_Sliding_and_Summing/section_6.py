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
        lecture_lines = [
            "Remember the workflow: Flip the filter, shift, and sum.",
            "Convolution is the language of linear systems and signals.",
            "It transforms raw data into meaningful, processed information."
        ]
        self.setup_layout("Summary and Real-World Impact", lecture_lines)
        
        # Colors for highlights
        COLOR_1 = "#00FFFF"  # Cyan
        COLOR_2 = "#00FF00"  # Green
        COLOR_3 = "#FFFF00"  # Yellow

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Flip Icon (Mirror)
        flip_tri_1 = Triangle(color=COLOR_1).scale(0.3)
        flip_tri_2 = flip_tri_1.copy().flip(RIGHT)
        flip_mirror_line = Line(UP, DOWN, color=WHITE).scale(0.4)
        flip_icon = VGroup(flip_tri_1, flip_mirror_line, flip_tri_2).arrange(RIGHT, buff=0.1)
        self.place_at_grid(flip_icon, "B2")
        
        # Shift Icon (Arrow) - Adjusted scale per Issue 30
        shift_icon = Arrow(start=LEFT, end=RIGHT, color=COLOR_1)
        self.place_at_grid(shift_icon, "B4", scale_factor=0.9)
        
        # Sum Icon (Sigma) - Adjusted scale per Issue 29
        sum_icon = MathTex(r"\sum", color=COLOR_1)
        self.place_at_grid(sum_icon, "B6", scale_factor=1.1)
        
        self.play(Flash(flip_icon, color=COLOR_1), FadeIn(flip_icon))
        self.play(Flash(shift_icon, color=COLOR_1), FadeIn(shift_icon))
        self.play(Flash(sum_icon, color=COLOR_1), FadeIn(sum_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Simplified CNN Diagram
        def create_layer(nodes_count, pos_col):
            layer = VGroup(*[Circle(radius=0.15, color=WHITE, fill_opacity=0.5) for _ in range(nodes_count)])
            layer.arrange(DOWN, buff=0.3)
            # Use middle rows for layers
            self.place_in_area(layer, f"D{pos_col}", f"F{pos_col}")
            return layer

        layer1 = create_layer(3, 2)
        layer2 = create_layer(4, 4)
        layer3 = create_layer(2, 6)
        
        connections = VGroup()
        for l1 in layer1:
            for l2 in layer2:
                connections.add(Line(l1.get_center(), l2.get_center(), stroke_width=1, color=COLOR_2))
        for l2 in layer2:
            for l3 in layer3:
                connections.add(Line(l2.get_center(), l3.get_center(), stroke_width=1, color=COLOR_2))
        
        self.play(Create(layer1), Create(layer2), Create(layer3))
        self.play(Create(connections))
        
        # Pulsing effect for connections
        pulse_tracker = ValueTracker(1)
        connections.add_updater(lambda m: m.set_stroke(width=pulse_tracker.get_value() * 2))
        
        self.play(pulse_tracker.animate.set_value(2), run_time=1, rate_func=there_and_back)
        self.play(pulse_tracker.animate.set_value(2), run_time=1, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Clear previous icons to make space
        self.play(
            FadeOut(flip_icon), FadeOut(shift_icon), FadeOut(sum_icon), 
            FadeOut(layer1), FadeOut(layer2), FadeOut(layer3), FadeOut(connections)
        )
        
        # Smartphone [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg] - Issue 20
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg", color=WHITE)
        # Position and Scale - Issue 31
        self.place_in_area(phone, "E5", "F6", scale_factor=0.8)
        
        # Rapid equations surrounding phone at E5-F6
        eqs = [
            MathTex(r"(f * g)[n]", font_size=18, color=COLOR_3),
            MathTex(r"\sum_{k} f[k]g[n-k]", font_size=16, color=COLOR_3),
            MathTex(r"y[n] = x[n] * h[n]", font_size=18, color=COLOR_3),
            MathTex(r"H(z) = \sum h[n]z^{-n}", font_size=16, color=COLOR_3)
        ]
        
        # Surround phone position (E5-F6) with equations
        grid_positions = ["D5", "D6", "E4", "F4"]
        for i, eq in enumerate(eqs):
            self.place_at_grid(eq, grid_positions[i])
            
        self.play(FadeIn(phone))
        
        # Pop in and out equations
        for eq in eqs:
            self.play(FadeIn(eq, scale=1.2), run_time=0.3)
            self.play(FadeOut(eq), run_time=0.3)
            
        self.wait(2)
