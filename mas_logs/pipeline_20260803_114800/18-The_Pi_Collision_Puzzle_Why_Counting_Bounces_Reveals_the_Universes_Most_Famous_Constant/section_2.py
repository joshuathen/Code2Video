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
        # Data from storyboard
        title = "Prerequisite Knowledge: The Laws of the Game"
        lines = [
            "Energy conservation keeps the system's total speed restricted.",
            "Momentum conservation governs how velocity transfers between blocks.",
            "These physical laws define the rules of every hit."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        GREEN = "#00FF00"
        BLUE = "#0000FF"

        # === Animation for Lecture Line 1 ===
        # Display 'Energy Conservation' in green (#00FF00) and 'Momentum Conservation' in blue (#0000FF)
        energy_label = Text("Energy Conservation", color=GREEN, font_size=24)
        momentum_label = Text("Momentum Conservation", color=BLUE, font_size=24)
        
        # Applying fixes from issues 35 and 36
        self.place_in_area(energy_label, 'B1', 'B2', scale_factor=0.8)
        self.place_in_area(momentum_label, 'D2', 'D5', scale_factor=0.8)

        self.lecture[0].set_color(GREEN)
        self.play(FadeIn(energy_label), FadeIn(momentum_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a pulsing green battery icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg]
        # Resolving Issue 28 and 37
        battery = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg").set_color(GREEN)
        self.place_in_area(battery, 'B3', 'C5', scale_factor=1.0)
        
        # Pulse mechanism using ValueTracker for performance (Instruction 10)
        pulse_tracker = ValueTracker(0)
        def pulse_updater(m):
            # Scale factor oscillating between 0.95 and 1.05
            val = pulse_tracker.get_value()
            scale = 1 + 0.05 * np.sin(val * PI * 2)
            # Update scale in-place to avoid expensive re-creation
            m.scale(scale / getattr(m, "pulse_prev_scale", 1.0))
            m.pulse_prev_scale = scale
            
        battery.pulse_prev_scale = 1.0
        battery.add_updater(pulse_updater)
        
        self.lecture[1].set_color(GREEN)
        self.play(FadeIn(battery))
        self.play(pulse_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw blue directional arrows (#0000FF) between the blocks during a collision
        block_large = Square(side_length=1.0, color=WHITE, fill_opacity=0.3)
        block_small = Square(side_length=0.5, color=WHITE, fill_opacity=0.3)
        
        self.place_at_grid(block_large, "E2", scale_factor=1.0)
        self.place_at_grid(block_small, "E5", scale_factor=1.0)
        
        # Momentum Arrow
        arrow = Arrow(start=self.grid["E2"], end=self.grid["E5"], color=BLUE, buff=0.2)
        
        self.lecture[2].set_color(BLUE)
        self.play(FadeIn(block_large), FadeIn(block_small))
        self.play(GrowArrow(arrow))
        
        # Simulate transfer: move arrow or flip it
        arrow_rev = Arrow(start=self.grid["E5"], end=self.grid["E2"], color=BLUE, buff=0.2)
        self.play(Transform(arrow, arrow_rev))
        self.wait(2)
        
        # Clean up updaters
        battery.remove_updater(pulse_updater)
