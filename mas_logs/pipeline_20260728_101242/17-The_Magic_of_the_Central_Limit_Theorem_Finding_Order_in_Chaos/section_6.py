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
        # Section Title and Lecture Lines from Storyboard
        lecture_lines = [
            "The CLT is the foundation of modern statistics.",
            "We can predict populations from just a few samples.",
            "Science and industry rely on this mathematical magic."
        ]
        self.setup_layout("Real-World Application", lecture_lines)
        
        # Paths for assets
        battery_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg"
        factory_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg"

        # Initialize lecture colors to dimmed state
        for line in self.lecture:
            line.set_color("#666666")

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)

        # Display silver (#C0C0C0) factory icon [Asset: factory.svg]
        factory = SVGMobject(factory_path).set_color("#C0C0C0")
        self.place_at_grid(factory, "A1", scale_factor=0.6)
        
        factory_label = Text("Production Plant", font_size=18, color="#C0C0C0")
        self.place_at_grid(factory_label, "A2", scale_factor=1.0)

        # Represent large production with a grid of battery icons [Asset: battery.svg]
        batteries = VGroup()
        for r in ["C", "D", "E"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                bat = SVGMobject(battery_path).set_color("#C0C0C0")
                self.place_at_grid(bat, f"{r}{c}", scale_factor=0.3)
                batteries.add(bat)

        pop_label = Text("Population (N = 1,000,000)", font_size=18, color="#C0C0C0")
        # FIX for Issue 39: Move to bottom row area for better centering
        self.place_in_area(pop_label, "F1", "F6", scale_factor=0.8)

        self.play(FadeIn(factory), Write(factory_label))
        self.play(LaggedStart(*[FadeIn(bat) for bat in batteries], lag_ratio=0.03))
        self.play(Write(pop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line, dim first
        self.play(
            self.lecture[0].animate.set_color("#666666"),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Highlight a sample group in yellow (#FFFF00)
        # Select some batteries to represent n=30
        sample_indices = [7, 8, 9, 13, 14, 15]
        sample_group = VGroup(*[batteries[i] for i in sample_indices])
        
        sample_label = Text("Sample (n = 30)", font_size=18, color="#FFFF00")
        # FIX for Issue 38: Move to B1 to avoid clutter with factory label at A2
        self.place_at_grid(sample_label, "B1", scale_factor=0.8)
        
        self.play(
            sample_group.animate.set_color("#FFFF00"),
            Write(sample_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line, dim second
        self.play(
            self.lecture[1].animate.set_color("#666666"),
            self.lecture[2].animate.set_color("#00FF00"),
            run_time=0.5
        )
        
        # Draw a green (#00FF00) bell curve to represent quality confidence
        axes = Axes(
            x_range=[-3, 3],
            y_range=[0, 1.2],
            x_length=3.0,
            y_length=1.5,
            axis_config={"include_tip": False, "include_ticks": False}
        ).set_color("#00FF00")
        
        # Normal distribution curve formula
        bell_curve = axes.plot(lambda x: np.exp(-x**2), color="#00FF00")
        curve_group = VGroup(axes, bell_curve)
        
        # Position curve in top right area
        self.place_in_area(curve_group, "A4", "B6", scale_factor=1.0)
        
        confidence_label = Text("Quality Confidence", font_size=18, color="#00FF00")
        # FIX for Issue 37: Move label to C5 to avoid overlapping the bell curve peak
        self.place_at_grid(confidence_label, "C5", scale_factor=0.8)
        
        self.play(
            Create(bell_curve),
            FadeIn(axes),
            Write(confidence_label)
        )
        self.wait(3)
