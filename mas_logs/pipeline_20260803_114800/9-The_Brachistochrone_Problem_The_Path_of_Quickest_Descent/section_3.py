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

class Section3Scene(TeachingScene):
    def construct(self):
        title_text = "The Optical Analogy (Snell's Law)"
        lecture_lines = [
            "Johann Bernoulli used an optical analogy to solve it.",
            "Imagine light traveling through layers with changing refractive indices.",
            "Snell’s Law states sin theta over velocity remains constant.",
            "Light always chooses the path of least time.",
            "This bending light path reveals the optimal physical curve."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for lines
        colors = ["#87CEEB", "#98FB98", "#F0E68C", "#FFA07A", "#DDA0DD"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Layers of varying density
        # Representing media where light speed changes
        # Varying colors (#1A1A1A to #4D4D4D)
        layer_colors = ["#1A1A1A", "#242424", "#2E2E2E", "#383838", "#424242", "#4D4D4D"]
        layers = VGroup()
        for i in range(6):
            row_char = chr(65 + i)
            # Area is from col 1 to 6 in each row
            rect = Rectangle(width=6.0, height=1.0, fill_color=layer_colors[i], fill_opacity=0.6, stroke_width=0.5, stroke_color=GREY_E)
            self.place_in_area(rect, f"{row_char}1", f"{row_char}6")
            layers.add(rect)
        
        self.play(FadeIn(layers))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Bending light path (Piecewise linear segments)
        # Light beam (#FFFF00)
        path_points = ["A2", "B3", "C4", "D5", "E6", "F6"]
        light_path = VGroup()
        for i in range(len(path_points)-1):
            line = Line(self.grid[path_points[i]], self.grid[path_points[i+1]], color="#FFFF00", stroke_width=4)
            light_path.add(line)
        
        for segment in light_path:
            self.play(Create(segment), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Snell's Law Formula: sin(theta)/v = constant in yellow (#F0E68C)
        # Issue 31 Fix: place_in_area(snell_formula, 'A4', 'B6', scale_factor=0.7)
        snell_formula = MathTex(r"\frac{\sin(\theta)}{v} = \text{constant}", color="#F0E68C")
        self.place_in_area(snell_formula, 'A4', 'B6', scale_factor=0.7)
        self.play(Write(snell_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # Transition to Lifeguard path analogy
        self.play(FadeOut(layers), FadeOut(light_path), FadeOut(snell_formula))
        
        # Beach and Water setup
        beach = Rectangle(width=6, height=3, fill_color="#EDC9AF", fill_opacity=0.8, stroke_width=0)
        self.place_in_area(beach, "A1", "C6")
        water = Rectangle(width=6, height=3, fill_color="#0077BE", fill_opacity=0.6, stroke_width=0)
        self.place_in_area(water, "D1", "F6")
        
        sand_label = Text("SAND (FAST)", font_size=18, color=BLACK)
        water_label = Text("WATER (SLOW)", font_size=18, color=WHITE)
        
        # Issue 32 & 33 Fixes: sand_label at 'C6', water_label at 'F6'
        self.place_at_grid(sand_label, 'C6')
        self.place_at_grid(water_label, 'F6')
        
        self.play(FadeIn(beach), FadeIn(water), Write(sand_label), Write(water_label))
        
        # Lifeguard optimal path (bending towards the normal when entering water)
        # Lifeguard path (#FF4500)
        lg_start = self.grid["A1"]
        lg_mid = self.grid["C4"] 
        lg_end = self.grid["F5"]
        
        lg_path = VGroup(
            Line(lg_start, lg_mid, color="#FF4500", stroke_width=4),
            Line(lg_mid, lg_end, color="#FF4500", stroke_width=4)
        )
        
        self.play(Create(lg_path), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        # Transformation into a smooth optimal curve (Brachistochrone/Cycloid)
        # Smooth Brachistochrone curve (#00FFFF)
        smooth_curve = VMobject(color="#00FFFF", stroke_width=6)
        # Defining curve points through grid anchors to keep it smooth
        smooth_curve.set_points_as_corners([self.grid["A1"], self.grid["B2"], self.grid["C4"], self.grid["E5"], self.grid["F6"]])
        smooth_curve.make_smooth()
        
        self.play(
            FadeOut(beach), FadeOut(water), FadeOut(sand_label), FadeOut(water_label),
            ReplacementTransform(lg_path, smooth_curve)
        )
        self.wait(2)
