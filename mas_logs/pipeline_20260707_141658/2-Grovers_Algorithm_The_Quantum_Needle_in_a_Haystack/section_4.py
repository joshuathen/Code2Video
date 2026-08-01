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
        # Data
        title_text = "Step 2: The Diffusion Operator (Inversion about the Mean)"
        lecture_lines = [
            "Calculate the average height of all amplitudes.",
            "Reflect every amplitude across this average line.",
            "The negative target amplitude shoots up high.",
            "Wrong answers shrink closer to zero height.",
            "This is the power of amplitude amplification."
        ]
        
        # Colors
        ORANGE = "#FF8C00"
        YELLOW = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"

        # Setup
        self.setup_layout(title_text, lecture_lines)

        # Positioning logic
        # We use a local coordinate system where y=0 is the mean line.
        # Amplitudes are measured from a baseline.
        # Average of amplitudes 1, 1, 1, 1, -1, 1, 1, 1 is 0.75.
        # Shifted so mean is at 0:
        # Original amplitudes: 1, 1, 1, 1, -1, 1, 1, 1
        # Baseline = -0.75
        # Initial tops: 0.25 (for amp 1), -1.75 (for amp -1)
        
        def create_bar_vgroup(amplitudes, baseline_y, target_index=4):
            group = VGroup()
            for i, amp in enumerate(amplitudes):
                color = YELLOW if i == target_index else WHITE_COLOR
                # Draw bar from baseline to top
                top_y = baseline_y + amp
                bottom_y = baseline_y
                rect = Rectangle(
                    width=0.4, 
                    height=abs(top_y - bottom_y), 
                    fill_opacity=0.8, 
                    color=color, 
                    stroke_width=2,
                    fill_color=color
                )
                # Position center of rect at midpoint
                rect.move_to([i * 0.6, (bottom_y + top_y) / 2, 0])
                group.add(rect)
            
            # Add phantom points to keep the y=0 (mean line) at the center of the VGroup
            # The maximum extent is from -1.75 to 0.25 (initial) or -0.75 to 1.75 (reflected)
            # So we use +/- 1.75 as bounds.
            phantom_top = Dot(point=[0, 1.75, 0], fill_opacity=0)
            phantom_bottom = Dot(point=[0, -1.75, 0], fill_opacity=0)
            group.add(phantom_top, phantom_bottom)
            return group

        # Initial amplitudes
        initial_amps = [1, 1, 1, 1, -1, 1, 1, 1]
        bars_group = create_bar_vgroup(initial_amps, -0.75)
        
        # Fix bars_group boundaries: Update to area A1-E5, scale 0.8
        self.place_in_area(bars_group, 'A1', 'E5', scale_factor=0.8)
        
        self.add(bars_group)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # "Calculate the average height of all amplitudes."
        self.play(self.lecture[0].animate.set_color(ORANGE))
        
        # Anchor 'mean_line': Update to area C1-C5, scale 1.0
        mean_line = DashedLine(
            start=LEFT * 2.5,
            end=RIGHT * 2.5,
            color=ORANGE,
            stroke_width=4
        )
        self.place_in_area(mean_line, 'C1', 'C5', scale_factor=1.0)
        
        # Adjust 'avg_label' scale: Update to grid C6, scale 0.6
        avg_label = Text("Average", font_size=24, color=ORANGE)
        self.place_at_grid(avg_label, 'C6', scale_factor=0.6)

        self.play(Create(mean_line), Write(avg_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Reflect every amplitude across this average line."
        self.play(self.lecture[1].animate.set_color(ORANGE))
        
        # Reflected amplitudes: 0.5 for wrong, 2.5 for target
        reflected_amps = [0.5, 0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5]
        reflected_bars_group = create_bar_vgroup(reflected_amps, -0.75)
        self.place_in_area(reflected_bars_group, 'A1', 'E5', scale_factor=0.8)

        self.play(
            ReplacementTransform(bars_group, reflected_bars_group),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The negative target amplitude shoots up high."
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Target bar is index 4 in the VGroup
        target_bar = reflected_bars_group[4]
        self.play(
            target_bar.animate.set_fill(opacity=1.0).scale(1.1, about_edge=DOWN),
            run_time=0.8
        )
        self.play(target_bar.animate.scale(1/1.1, about_edge=DOWN), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Wrong answers shrink closer to zero height."
        self.play(self.lecture[3].animate.set_color(WHITE_COLOR))
        
        wrong_bars = VGroup(*[reflected_bars_group[i] for i in range(8) if i != 4])
        self.play(
            wrong_bars.animate.set_fill(opacity=0.3),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This is the power of amplitude amplification."
        self.play(self.lecture[4].animate.set_color(WHITE_COLOR))
        
        # Highlight success with a flash at the peak of the target bar
        flash_point = target_bar.get_top()
        flash = Circle(radius=0.1, color=WHITE_COLOR, fill_opacity=1, stroke_width=0).move_to(flash_point)
        self.play(
            flash.animate.scale(20).set_opacity(0),
            run_time=1.2,
            rate_func=exponential_decay
        )
        self.wait(2)
