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

class Section2Scene(TeachingScene):
    def construct(self):
        # Set up title and lecture lines
        title_text = "Prerequisite: Signals as Sequences"
        lecture_lines = [
            "Discrete signals are simple sequences of numerical values.",
            "We have an input signal and a filter kernel.",
            "Signals are mapped to integer indices on a graph."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        GREEN = "#00FF00"
        YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: "Discrete signals are simple sequences of numerical values."
        # Animation: Show a horizontal axis with vertical bars representing [1, 2, 3, 2, 1] in #FFFFFF.
        self.lecture[0].set_color(WHITE_COLOR)
        
        # Horizontal axis for x[n] at row D
        # Start at col 2, end at col 6 to utilize more of the right side (Issue 25)
        x_axis = Line(self.grid["D2"] + LEFT*0.5, self.grid["D6"] + RIGHT*0.5, color=WHITE_COLOR)
        
        x_values = [1, 2, 3, 2, 1]
        x_bars = VGroup()
        for i, val in enumerate(x_values):
            col_idx = i + 2 # Start from col 2
            col = str(col_idx)
            start = self.grid[f"D{col}"]
            
            # Row D is y=-0.8, C is y=0.2, B is y=1.2, A is y=2.2
            if val == 1: target_row = "C"
            elif val == 2: target_row = "B"
            else: target_row = "A" # val 3
            
            end = self.grid[f"{target_row}{col}"]
            bar = Line(start, end, color=WHITE_COLOR, stroke_width=8)
            dot = Dot(end, color=WHITE_COLOR, radius=0.1)
            x_bars.add(VGroup(bar, dot))
            
        self.play(Create(x_axis), Create(x_bars), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "We have an input signal and a filter kernel."
        # Animation: Label the sequence "x[n]" in green #00FF00 above the bars.
        self.lecture[1].set_color(GREEN)
        
        # Sequence label x[n] at A4 (Issue 25)
        x_label = MathTex("x[n]", color=GREEN)
        self.place_at_grid(x_label, "A4")
        x_label.shift(UP * 0.6)
        
        self.play(Write(x_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "Signals are mapped to integer indices on a graph."
        # Animation: Show a smaller sequence [0.5, 1, 0.5] labeled "h[n]" in yellow #FFFF00 below.
        self.lecture[2].set_color(YELLOW)
        
        # Index labels for the x_axis
        x_indices = VGroup()
        for i in range(5):
            idx = Text(str(i), font_size=18, color=WHITE)
            col_idx = i + 2
            self.place_at_grid(idx, f"D{col_idx}")
            idx.shift(DOWN * 0.3)
            x_indices.add(idx)

        # Horizontal axis for h[n] at row F, columns 4 to 6
        h_axis = Line(self.grid["F4"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=WHITE_COLOR)
        
        h_values = [0.5, 1.0, 0.5]
        h_bars = VGroup()
        for i, val in enumerate(h_values):
            col_idx = i + 4
            col = str(col_idx)
            start = self.grid[f"F{col}"]
            # Row F is -2.8, E is -1.8
            if val == 1.0:
                end = self.grid[f"E{col}"]
            else:
                # 0.5 is mid point between F and E
                end = (self.grid[f"F{col}"] + self.grid[f"E{col}"]) / 2
            
            bar = Line(start, end, color=YELLOW, stroke_width=6)
            dot = Dot(end, color=YELLOW, radius=0.08)
            h_bars.add(VGroup(bar, dot))
            
        # h_label at E5 (Issue 24)
        h_label = MathTex("h[n]", color=YELLOW)
        self.place_at_grid(h_label, "E5")
        h_label.shift(UP * 0.5)
        
        self.play(
            Write(x_indices),
            Create(h_axis), 
            Create(h_bars), 
            Write(h_label),
            run_time=2
        )
        self.wait(2)
