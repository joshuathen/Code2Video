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
        # Initial Setup
        title = "Step 2: The Diffusion Operator (Amplitude Amplification)"
        lines = [
            "We calculate the average height of all amplitudes.",
            "Every wave reflects across this average line.",
            "The flipped target state is pushed significantly higher.",
            "Remaining non-target amplitudes shrink in the process.",
            "This makes the correct answer stand out clearly."
        ]
        self.setup_layout(title, lines)

        # Define layout constants
        BASELINE_Y = self.grid["D1"][1]  # Row D
        X_START = 1.0
        X_END = 5.0
        BAR_WIDTH = 0.4
        NUM_BARS = 8
        TARGET_IDX = 3
        
        # Initial Heights
        h_init_others = 0.8
        h_init_target = -0.8
        
        # Colors
        COLOR_OTHER = "#0000FF" # Blue
        COLOR_TARGET = "#FFA500" # Orange
        COLOR_MEAN = "#00FFFF" # Cyan
        COLOR_TEXT = "#FFFF00" # Yellow

        # Create Baseline
        baseline = Line(
            start=[X_START - 0.5, BASELINE_Y, 0],
            end=[X_END + 0.5, BASELINE_Y, 0],
            color=WHITE,
            stroke_width=2
        )
        self.add(baseline)

        # Helper to create a bar at height
        def get_bar(index, height):
            color = COLOR_TARGET if index == TARGET_IDX else COLOR_OTHER
            x_pos = X_START + index * (X_END - X_START) / (NUM_BARS - 1)
            bar = Rectangle(
                width=BAR_WIDTH,
                height=abs(height),
                fill_color=color,
                fill_opacity=0.8,
                stroke_width=1,
                stroke_color=WHITE
            )
            # Center of rectangle should be at height/2 relative to baseline
            bar.move_to([x_pos, BASELINE_Y + height/2, 0])
            return bar

        # Create Initial Bars
        bars = VGroup(*[get_bar(i, h_init_others if i != TARGET_IDX else h_init_target) for i in range(NUM_BARS)])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(bars))
        
        # Calculate Average
        avg_h = ((NUM_BARS - 1) * h_init_others + h_init_target) / NUM_BARS
        mean_line_y = BASELINE_Y + avg_h
        mean_line = Line(
            start=[X_START - 0.5, mean_line_y, 0],
            end=[X_END + 0.5, mean_line_y, 0],
            color=COLOR_MEAN,
            stroke_width=4
        )
        mean_label = Text("Average", font_size=16, color=COLOR_MEAN)
        # Fix for Issue 40: Move mean_label to C5 and avoid manual positioning
        self.place_at_grid(mean_label, "C5", scale_factor=0.8)

        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Calculate New Heights
        h_final_others = 2 * avg_h - h_init_others
        h_final_target = 2 * avg_h - h_init_target
        
        # Prepare Target Bars for Transformation
        final_bars = VGroup(*[get_bar(i, h_final_others if i != TARGET_IDX else h_final_target) for i in range(NUM_BARS)])
        
        self.play(
            ReplacementTransform(bars, final_bars),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight the tall bar
        target_bar_final = final_bars[TARGET_IDX]
        rect_highlight = SurroundingRectangle(target_bar_final, color=COLOR_TARGET, buff=0.1)
        
        self.play(Create(rect_highlight))
        self.play(Indicate(target_bar_final, color=COLOR_TARGET))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight shrinking others
        other_bars = VGroup(*[final_bars[i] for i in range(NUM_BARS) if i != TARGET_IDX])
        self.play(FadeOut(rect_highlight))
        self.play(LaggedStart(*[Indicate(b, color=COLOR_OTHER) for b in other_bars], lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        standout_text = Text("Amplitude Amplification: Target stands out", color=COLOR_TEXT, font_size=20)
        # Fix for Issue 41: Move standout_text to area E1-E6 and adjust scale
        self.place_in_area(standout_text, "E1", "E6", scale_factor=0.8)
        
        self.play(Write(standout_text))
        self.wait(2)
        
        # Clean up for transition
        self.lecture[4].set_color(WHITE)
        self.wait(1)
