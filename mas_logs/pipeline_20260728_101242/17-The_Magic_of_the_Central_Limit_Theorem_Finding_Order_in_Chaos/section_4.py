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
        title = "The Visual Transformation"
        lecture_lines = [
            "Watch the distribution of these sample means change.",
            "Initially, the averages might look a bit scattered.",
            "As we collect more samples, a pattern emerges.",
            "The messy distribution morphs into a smooth curve.",
            "It becomes the famous, symmetrical Bell Curve."
        ]
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show the original messy rectangular histogram dimmed in the background.
        self.lecture[0].set_color(YELLOW)
        
        # Creating a placeholder "messy" histogram using bars
        heights = [1.2, 0.8, 1.5, 0.5, 1.1, 1.4, 0.7, 1.3]
        bars = VGroup(*[Rectangle(width=0.4, height=h, fill_opacity=0.3, color=GRAY, stroke_width=1) for h in heights])
        bars.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Fix Issue 33: Move bars to B3-E6 to avoid cramping on the left
        self.place_in_area(bars, "B3", "E6", scale_factor=0.8)
        
        # Add a baseline
        baseline = Line(bars.get_left(), bars.get_right(), color=WHITE).shift(DOWN * 0.05)
        
        self.play(FadeIn(bars), Create(baseline))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot new sample mean points in yellow (#FFFF00) along the horizontal axis.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        np.random.seed(42)
        num_dots = 30
        dots = VGroup()
        # center_x is calculated based on the updated bars position
        center_x = bars.get_center()[0]
        base_y = baseline.get_y() + 0.1
        
        # Generate random positions clustered in the middle
        for _ in range(num_dots):
            # Cluster around center_x
            x_offset = np.random.normal(0, 0.8) 
            dot = Dot(point=[center_x + x_offset, base_y, 0], radius=0.06, color="#FFFF00")
            dots.add(dot)

        self.play(Create(dots), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Morph the accumulating points into a rough light blue (#ADD8E6) bell shape.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Create a rough bell shape using points
        rough_points = [
            [-2.0, 0.0, 0], [-1.5, 0.4, 0], [-1.0, 1.2, 0], [-0.5, 2.0, 0], 
            [0.0, 2.5, 0], [0.5, 1.9, 0], [1.0, 1.3, 0], [1.5, 0.5, 0], [2.0, 0.0, 0]
        ]
        rough_curve = VMobject(color="#ADD8E6", stroke_width=4)
        rough_curve.set_points_as_corners([np.array(p) for p in rough_points])
        
        # Fix Issue 31: Consistent scale_factor=0.8 and position B3-E6
        self.place_in_area(rough_curve, "B3", "E6", scale_factor=0.8)
        # Ensure it aligns with the baseline for consistency
        rough_curve.align_to(baseline, DOWN)
        
        self.play(
            FadeOut(dots),
            FadeOut(bars),
            Create(rough_curve),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Smooth the rough shape into a perfect pink (#FFC0CB) normal distribution curve.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Use an Axes to generate the smooth curve
        ax = Axes(
            x_range=[-3, 3], 
            y_range=[0, 0.5], 
            x_length=5, 
            y_length=3,
            axis_config={"include_tip": False}
        )
        # Fix Issue 32: Place ax in area E3-E6 so it sits at the base
        self.place_in_area(ax, "E3", "E6", scale_factor=0.8)
        
        smooth_curve = ax.plot(
            lambda x: 0.4 * np.exp(-0.5 * x**2),
            color="#FFC0CB",
            stroke_width=5
        )
        # Ensure it aligns with the baseline for consistency
        smooth_curve.align_to(baseline, DOWN)

        self.play(
            Transform(rough_curve, smooth_curve),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the final curve to emphasize the completed transformation.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Indicate the transformed curve (stored in rough_curve reference)
        self.play(
            Indicate(rough_curve, color="#FFC0CB", scale_factor=1.2),
            run_time=1.5
        )
        self.wait(2)
