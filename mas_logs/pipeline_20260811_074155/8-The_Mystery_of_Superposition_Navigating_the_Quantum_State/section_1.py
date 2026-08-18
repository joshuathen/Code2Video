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
        # Define missing color
        CYAN = "#00FFFF"
        
        lecture_lines = [
            "Classical objects exist in one definite state at once.",
            "A spinning coin represents a blur of possible states.",
            "Quantum particles exist in this \"blur\" as their reality."
        ]
        self.setup_layout("The Boundary: Classical vs. Quantum", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a binary switch flipping between '0' and '1' in white (#FFFFFF).
        self.lecture[0].set_color(WHITE)
        
        switch_base = RoundedRectangle(width=3, height=1, corner_radius=0.5, color=WHITE)
        # Shifted to area C3-D6 (Issue 29, 42)
        self.place_in_area(switch_base, "C3", "D6")
        
        label_0 = Text("0", font_size=36, color=WHITE)
        label_1 = Text("1", font_size=36, color=WHITE)
        # Shifted to C3 and C6 (Issue 30, 42)
        self.place_at_grid(label_0, "C3")
        self.place_at_grid(label_1, "C6")
        
        knob = Circle(radius=0.4, color=WHITE, fill_opacity=1)
        knob.move_to(label_0.get_center())
        
        switch_group = VGroup(switch_base, label_0, label_1, knob)
        self.play(FadeIn(switch_group))
        
        # Flip back and forth
        self.play(knob.animate.move_to(label_1.get_center()), run_time=0.8)
        self.wait(0.2)
        self.play(knob.animate.move_to(label_0.get_center()), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the switch into a blurry, rotating cloud in cyan (#00FFFF).
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(CYAN)
        
        # Create a blurry cloud using many small dots
        np.random.seed(42)
        num_dots = 100
        cloud_dots = VGroup(*[
            Dot(
                point=[np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0],
                radius=0.08,
                color=CYAN,
                fill_opacity=np.random.uniform(0.1, 0.5)
            )
            for _ in range(num_dots)
        ])
        # Shifted to area C3-D6 (Issue 29, 42)
        self.place_in_area(cloud_dots, "C3", "D6")
        
        # Setup rotation via updater to avoid always_redraw overhead
        cloud_dots.add_updater(lambda d, dt: d.rotate(0.05))

        self.play(
            ReplacementTransform(switch_group, cloud_dots),
            run_time=1.5
        )
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        # Display the text 'SUPERPOSITION' in bold cyan (#00FFFF) above the cloud.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(CYAN)
        
        superposition_text = Text("SUPERPOSITION", font_size=40, color=CYAN, weight=BOLD)
        # Shifted to area B4-B5 (Issue 31, 42)
        self.place_in_area(superposition_text, "B4", "B5")
        
        self.play(Write(superposition_text))
        
        self.wait(3)
