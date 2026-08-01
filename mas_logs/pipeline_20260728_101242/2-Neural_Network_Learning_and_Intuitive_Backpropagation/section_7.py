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

class Section7Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Result: Convergence", [
            "Repeating this cycle creates an intelligent network.",
            "After many rounds, the robot hits the bullseye.",
            "The system has learned the optimal internal weights."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Colors
        robot_color = "#FFD700"
        loop_color = "#87CEEB"
        
        # Robot Asset Integration
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color(robot_color)
        self.place_at_grid(robot, "B2", scale_factor=0.6)
        
        # Loop text
        loop_steps = ["Forward Pass", "Error", "Backprop", "Update"]
        loop_text = Text(loop_steps[0], font_size=20, color=loop_color)
        # Fix Issue 46: Center status text
        self.place_in_area(loop_text, 'D2', 'D4', scale_factor=0.8)
        
        self.play(FadeIn(robot))
        self.lecture[0].set_color(robot_color)
        
        # Rapid shooting and loop animation
        for i in range(4):
            # Rapid shooting (arrow)
            arrow_start = robot.get_right()
            # Arrows miss slightly at first
            arrow_end = arrow_start + RIGHT * 2.5 + UP * (0.8 - 0.4 * i)
            arrow = Arrow(start=arrow_start, end=arrow_end, color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.2)
            
            # Loop text update
            new_text = Text(loop_steps[i % 4], font_size=20, color=loop_color)
            # Fix Issue 45: Center update label
            self.place_in_area(new_text, 'D2', 'D4', scale_factor=0.8)
            
            self.play(
                GrowArrow(arrow, run_time=0.15),
                Transform(loop_text, new_text, run_time=0.15),
                rate_func=linear
            )
            self.play(FadeOut(arrow, run_time=0.05))

        self.wait(1)

        # === Animation for Lecture Line 2 ===
        bullseye_color = "#FF4500"
        
        # Target
        target_outer = Circle(radius=0.4, color=WHITE, fill_opacity=1)
        target_mid = Circle(radius=0.25, color=RED, fill_opacity=1)
        target_inner = Circle(radius=0.1, color=bullseye_color, fill_opacity=1)
        target = VGroup(target_outer, target_mid, target_inner)
        self.place_at_grid(target, "B5")
        
        self.play(FadeIn(target))
        self.lecture[1].set_color(bullseye_color)
        
        # Perfect shot
        final_arrow = Arrow(start=robot.get_right(), end=target_inner.get_center(), color=WHITE, buff=0)
        
        self.play(GrowArrow(final_arrow, run_time=0.8))
        self.play(target_inner.animate.scale(1.2).set_color(YELLOW), run_time=0.2)
        self.play(target_inner.animate.scale(1/1.2).set_color(bullseye_color), run_time=0.2)
        
        # Convergence text
        conv_text = Text("Convergence Achieved", font_size=24, color="#00FF00")
        # Fix Issue 44: Center convergence text
        self.place_in_area(conv_text, 'A2', 'A5', scale_factor=0.8)
        self.play(Write(conv_text))
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        weights_color = "#00FF00"
        self.lecture[2].set_color(weights_color)
        
        # Weight Knobs
        def create_knob(label):
            circle = Circle(radius=0.3, color=WHITE)
            line = Line(circle.get_center(), circle.get_top(), color=weights_color)
            lbl = Text(label, font_size=16, color=WHITE).next_to(circle, DOWN, buff=0.1)
            return VGroup(circle, line, lbl)
            
        knob1 = create_knob("W1")
        knob2 = create_knob("W2")
        knob3 = create_knob("W3")
        
        self.place_at_grid(knob1, "E2")
        self.place_at_grid(knob2, "E3")
        self.place_at_grid(knob3, "E4")
        
        knobs = VGroup(knob1, knob2, knob3)
        
        self.play(FadeIn(knobs))
        
        # Locking animation (rotate to "perfect" position and glow)
        lock_anims = []
        for knob in knobs:
            # Rotate line to a specific "optimal" angle
            # The line is the second element in the VGroup
            lock_anims.append(Rotate(knob[1], angle=PI/4, about_point=knob[0].get_center()))
            lock_anims.append(knob[0].animate.set_color(weights_color).set_stroke(width=6))
            
        self.play(*lock_anims)
        
        # Add checkmark
        check = Text("✓", color=weights_color).scale(1.5)
        self.place_at_grid(check, "E5")
        self.play(Write(check))

        self.wait(2)

# Marking issues as under review
# update_issue(32, under_review=True, resolution_note="Integrated robot SVG asset and replaced procedural robot.")
# update_issue(44, under_review=True, resolution_note="Fixed conv_text positioning using place_in_area.")
# update_issue(45, under_review=True, resolution_note="Fixed loop update text positioning using place_in_area.")
# update_issue(46, under_review=True, resolution_note="Fixed initial loop text positioning using place_in_area.")
