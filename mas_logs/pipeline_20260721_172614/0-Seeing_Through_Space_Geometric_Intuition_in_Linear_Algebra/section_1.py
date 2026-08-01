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
        # Data from storyboard and outline
        title = "The Vector as an Arrow (0:00 - 0:45)"
        lecture_lines = [
            "Vectors are physical arrows pointing in 2D space.",
            "We can add them using the tip-to-tail method.",
            "Scalar multiplication stretches or squishes the arrow's length."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Grid setup for animation
        # Issue 27: Fix coordinate plane position and scale
        plane = NumberPlane(
            x_range=[-1, 7, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": "#444444", "stroke_opacity": 0.5},
            axis_config={"stroke_color": "#444444", "include_tip": True}
        )
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(plane))
        
        # Green arrow [3, 2]
        green_vec = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(3, 2),
            buff=0,
            color="#00FF00",
            stroke_width=6
        )
        
        # Issue 29: Fix: label [3,2] at C4
        label_a = MathTex("[3, 2]", color="#00FF00")
        self.place_at_grid(label_a, 'C4', scale_factor=0.5)
        
        self.play(GrowArrow(green_vec), Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#0000FF")
        )
        
        # Blue arrow [1, 2] attached tip-to-tail
        blue_vec = Arrow(
            start=plane.c2p(3, 2),
            end=plane.c2p(4, 4),
            buff=0,
            color="#0000FF",
            stroke_width=6
        )
        
        # Position blue label manually at a good grid spot
        label_b = MathTex("[1, 2]", color="#0000FF")
        self.place_at_grid(label_b, 'B5', scale_factor=0.5)
        
        # White sum arrow [4, 4]
        sum_vec = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(4, 4),
            buff=0,
            color="#FFFFFF",
            stroke_width=4
        )
        
        # Issue 28: Fix: label [4,4] at A5
        label_sum = MathTex("[4, 4]", color="#FFFFFF")
        self.place_at_grid(label_sum, 'A5', scale_factor=0.5)
        
        self.play(GrowArrow(blue_vec), Write(label_b))
        self.play(Create(sum_vec), Write(label_sum))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # Clear blue and white vectors to focus on scaling green
        self.play(
            FadeOut(blue_vec), FadeOut(label_b),
            FadeOut(sum_vec), FadeOut(label_sum)
        )
        
        # Scaling green vector from [3, 2] to [6, 4]
        # Prepare the target label
        label_a_scaled = MathTex("[6, 4]", color="#00FF00")
        self.place_at_grid(label_a_scaled, 'B6', scale_factor=0.5)
        
        scale_tracker = ValueTracker(1.0)
        
        # Update function for the green vector to follow the tracker
        def update_vec(v):
            val = scale_tracker.get_value()
            new_end = plane.c2p(3 * val, 2 * val)
            v.put_start_and_end_on(plane.c2p(0, 0), new_end)
            
        green_vec.add_updater(update_vec)
        
        # Animate scaling of arrow and transformation of label
        self.play(
            scale_tracker.animate.set_value(2.0),
            Transform(label_a, label_a_scaled),
            run_time=2
        )
        self.wait(1)
        
        green_vec.remove_updater(update_vec)
        
        self.wait(2)
