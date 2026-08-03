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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "Mathematical Visualization: The Dot Product Match"
        lines = [
            "Input vector x is compared to key vector ki.",
            "The dot product measures the similarity between them.",
            "High similarity causes the hidden neuron to fire strongly.",
            "This firing triggers the retrieval of the associated value.",
            "Facts are stored in the alignment of these vectors."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_X = "#FFFFFF"
        COLOR_KI = "#FF69B4"
        COLOR_GLOW = "#FF4500"
        COLOR_BAR = "#FFFF00"
        COLOR_VALUE = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        vector_x = Vector([1, 1, 0], color=COLOR_X)
        label_x = MathTex("x", color=COLOR_X).scale(0.8)
        self.place_at_grid(vector_x, "B3") # Resolved Issue 38
        label_x.next_to(vector_x, UP, buff=0.1)
        
        vector_ki = Vector([1, -0.5, 0], color=COLOR_KI)
        label_ki = MathTex("k_i", color=COLOR_KI).scale(0.8)
        self.place_at_grid(vector_ki, "B5") # Resolved Issue 39 (part 1)
        label_ki.next_to(vector_ki, UP, buff=0.1)
        
        self.play(Create(vector_x), Write(label_x))
        self.play(Create(vector_ki), Write(label_ki))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        formula = MathTex("x \\cdot k_i", color=WHITE)
        self.place_at_grid(formula, "B4", scale_factor=0.9) # Resolved Issue 39 (part 2)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Vectors rotate to align
        # Create a glow circle behind the alignment point
        glow = Arc(radius=0.8, start_angle=0, angle=TAU, color=COLOR_GLOW, stroke_width=0).set_fill(COLOR_GLOW, opacity=0.3)
        self.place_at_grid(glow, "B4") # Resolved Issue 39 (part 3)
        
        # Asset: Thermometer
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg").scale(0.5)
        self.place_at_grid(thermometer, "A4") # Positioned above the glow/formula
        
        self.play(
            vector_x.animate.rotate(angle=-vector_x.get_angle(), about_point=self.grid["B4"]),
            vector_ki.animate.rotate(angle=-vector_ki.get_angle(), about_point=self.grid["B4"]),
            FadeIn(glow, run_time=1.5),
            FadeIn(thermometer),
            formula.animate.scale(1.2),
            label_x.animate.next_to(self.grid["B4"], UP, buff=0.5),
            label_ki.animate.next_to(self.grid["B4"], DOWN, buff=0.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Bar graph "Activation"
        bar_base = self.grid["E2"]
        bar = Rectangle(height=0.1, width=0.8, color=COLOR_BAR, fill_opacity=0.8)
        bar.move_to(bar_base + UP * 0.05)
        bar_label = Text("Activation", font_size=18, color=COLOR_BAR)
        bar_label.next_to(bar, DOWN, buff=0.2)
        
        self.play(Create(bar), Write(bar_label))
        # Activation height increase
        self.play(
            bar.animate.stretch_to_fit_height(1.5, about_edge=DOWN),
            glow.animate.set_fill(opacity=0.6).scale(1.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Value vector retrieval
        value_vector = Vector([0, 2, 0], color=COLOR_VALUE)
        value_label = Text("Value", font_size=18, color=COLOR_VALUE)
        self.place_at_grid(value_vector, "E5") # Resolved Issue 40
        value_label.next_to(value_vector, RIGHT, buff=0.2)
        
        self.play(
            Create(value_vector),
            Write(value_label)
        )
        
        # Scale value vector by activation (symbolically)
        self.play(
            value_vector.animate.scale(1.4),
            value_label.animate.scale(1.2)
        )
        
        self.wait(2)
