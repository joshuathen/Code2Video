from manim import *

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
        # Data from shared state
        title_text = "Scalar Multiplication: Stretching and Flipping"
        lecture_lines = [
            "Multiplying a vector by a number scales its length.",
            "A multiplier greater than one stretches the vector.",
            "Between zero and one, the vector shrinks.",
            "Multiplying by a negative number flips the direction.",
            "The vector stays on the same line of action."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        GOLD_V = "#FFD700"
        RED_V = "#FF0000"
        WHITE_V = "#FFFFFF"
        GRAY_LINE = "#888888"
        
        # Animation Elements Setup
        scalar = ValueTracker(1.0)
        base_length = 1.2
        # REVISED: Start at C3 to avoid crowding lecture notes (Issues 29, 30, 31)
        start_point = self.grid["C3"]
        
        # Vector mobject using Arrow with buff=0
        vector = Arrow(
            start=start_point, 
            end=start_point + RIGHT * base_length, 
            buff=0, 
            color=GOLD_V,
            stroke_width=6
        )
        
        def vector_updater(mob):
            val = scalar.get_value()
            # Ensure a minimal render length to prevent Arrow head calculation errors at 0
            render_val = val if abs(val) > 0.001 else 0.001
            
            mob.put_start_and_end_on(start_point, start_point + RIGHT * render_val * base_length)
            mob.set_color(RED_V if val < 0 else GOLD_V)
            
        vector.add_updater(vector_updater)
        
        # Labels
        label_v = MathTex("v", color=WHITE_V)
        label_2v = MathTex("2v", color=WHITE_V)
        label_05v = MathTex("0.5v", color=WHITE_V)
        label_negv = MathTex("-v", color=WHITE_V)
        
        # REVISED: Position labels at D3 (below start_point C3) per Issues 29, 30, 31
        self.place_at_grid(label_v, "D3", scale_factor=0.8)
        self.place_at_grid(label_2v, "D3", scale_factor=0.8)
        self.place_at_grid(label_05v, "D3", scale_factor=0.8)
        self.place_at_grid(label_negv, "D3", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "Multiplying a vector by a number scales its length."
        self.lecture[0].set_color(GOLD_V)
        self.play(Create(vector), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A multiplier greater than one stretches the vector."
        self.lecture[1].set_color(GOLD_V)
        self.play(
            scalar.animate.set_value(2.0),
            FadeOut(label_v, shift=UP*0.1),
            FadeIn(label_2v, shift=UP*0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Between zero and one, the vector shrinks."
        self.lecture[2].set_color(GOLD_V)
        self.play(
            scalar.animate.set_value(0.5),
            FadeOut(label_2v, shift=UP*0.1),
            FadeIn(label_05v, shift=UP*0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Multiplying by a negative number flips the direction."
        self.lecture[3].set_color(RED_V)
        self.play(
            scalar.animate.set_value(-1.0),
            FadeOut(label_05v, shift=UP*0.1),
            FadeIn(label_negv, shift=UP*0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The vector stays on the same line of action."
        self.lecture[4].set_color(WHITE_V)
        # Line of action extends horizontally
        line_of_action = DashedLine(
            start=self.grid["C2"],
            end=self.grid["C6"],
            color=GRAY_LINE,
            stroke_opacity=0.6
        )
        
        # Ensure the vector is drawn on top of the line
        self.add(line_of_action)
        self.add(vector) # Moving to front layer
        
        self.play(Create(line_of_action))
        self.wait(3)
