from manim import *
import numpy as np
import random

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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup data from storyboard
        title_text = "The Experiment: Sampling the Squirrels"
        lecture_lines = [
            "We calculate the average weight of squirrel groups.",
            "Each sample mean becomes a single data point.",
            "Repeating this process creates a new data distribution."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        GREY_COLOR = "#808080"
        BLUE_COLOR = "#0000FF"
        
        # Static elements: Axis at the bottom (Row F)
        axis = Line(self.grid["F1"], self.grid["F6"], color=WHITE)
        axis_label = Text("Sample Mean Weight", font_size=16, color=WHITE).next_to(axis, DOWN, buff=0.2)
        self.add(axis, axis_label)

        # Assets
        squirrel_svg_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/squirrel.svg"

        # === Animation for Lecture Line 1 ===
        # Color line 1 to match elements (Grey/White context)
        self.play(self.lecture[0].animate.set_color(GREY_COLOR))
        
        # 10 squirrels representing individual squirrels (Asset Integration)
        squirrel_template = SVGMobject(squirrel_svg_path, color=GREY_COLOR, fill_opacity=0.8).scale(0.3)
        squirrels = VGroup(*[squirrel_template.copy() for _ in range(10)])
        squirrels.arrange_in_grid(rows=2, cols=5, buff=0.2)
        
        # Positioning according to Issue 40 fix
        self.place_in_area(squirrels, "C2", "D5", scale_factor=0.6)
        
        self.play(FadeIn(squirrels))
        self.wait(0.5)
        
        # Calculate average and display number above a blue dot
        avg_val = 502
        avg_label = Text(f"{avg_val}g", font_size=20, color=WHITE)
        sample_dot = Dot(color=BLUE_COLOR, radius=0.1)
        
        # Positioning according to Issue 41 fix
        self.place_at_grid(sample_dot, 'E3', scale_factor=0.8)
        avg_label.next_to(sample_dot, UP, buff=0.2)
        
        self.play(
            Write(avg_label),
            FadeIn(sample_dot)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2 to match elements (Blue)
        self.play(self.lecture[1].animate.set_color(BLUE_COLOR))
        
        # The blue dot falls vertically onto the axis
        # Position it on the axis (Row F)
        target_x = sample_dot.get_x()
        target_y = self.grid["F3"][1] + 0.1
        
        self.play(
            sample_dot.animate.move_to([target_x, target_y, 0]),
            FadeOut(avg_label),
            FadeOut(squirrels),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 to match elements (Blue)
        self.play(self.lecture[2].animate.set_color(BLUE_COLOR))
        
        # Stacking logic setup
        bin_counts = {}
        bin_width = 0.4
        dot_radius = 0.08
        
        def get_stack_pos(x_val):
            # Map x_val to a bin index based on the axis range
            # Range F1 to F6 is roughly from x=0.5 to x=5.5
            start_x = self.grid["F1"][0]
            bin_idx = int((x_val - start_x) / bin_width)
            count = bin_counts.get(bin_idx, 0)
            bin_counts[bin_idx] = count + 1
            # Stack dots upwards from the axis
            return np.array([start_x + bin_idx * bin_width + bin_width/2, self.grid["F1"][1] + 0.1 + count * (dot_radius * 1.5), 0])

        # Register the first dot in the bins
        first_bin_idx = int((target_x - self.grid["F1"][0]) / bin_width)
        bin_counts[first_bin_idx] = 1

        # Repeat the process a few times with visual squirrel flashes
        for _ in range(5):
            # Flash squirrel group (Issue 42 fix)
            temp_sq = VGroup(*[squirrel_template.copy() for _ in range(10)])
            temp_sq.arrange_in_grid(rows=2, cols=5, buff=0.1)
            self.place_in_area(temp_sq, "C2", "D5", scale_factor=0.6)
            
            # Generate a new sample mean dot
            # Sample around the center of the axis
            center_x = (self.grid["F1"][0] + self.grid["F6"][0]) / 2
            rand_x = center_x + random.uniform(-1.5, 1.5)
            new_dot = Dot(color=BLUE_COLOR, radius=dot_radius)
            # Start from the squirrel group area (Row C/D)
            new_dot.move_to([rand_x, self.grid["D3"][1], 0])
            
            self.play(
                FadeIn(temp_sq, run_time=0.2),
                FadeIn(new_dot, run_time=0.2)
            )
            
            # Dot falls and stacks
            stack_pos = get_stack_pos(rand_x)
            self.play(
                new_dot.animate.move_to(stack_pos),
                FadeOut(temp_sq, run_time=0.2),
                run_time=0.6
            )

        # Speed up the process (Rapidly stacking dots)
        fast_dots_animations = []
        num_fast_dots = 60
        for i in range(num_fast_dots):
            # Sampling from a normal distribution for visual effect (CLT demo)
            # The underlying squirrel weights are uniform, but sample means are normal
            center_x = (self.grid["F1"][0] + self.grid["F6"][0]) / 2
            rand_x = random.gauss(center_x, 0.7)
            # Clamp to axis bounds
            rand_x = max(self.grid["F1"][0] + 0.2, min(self.grid["F6"][0] - 0.2, rand_x))
            
            fast_dot = Dot(color=BLUE_COLOR, radius=dot_radius * 0.7)
            # Start falling from above (Row B)
            start_pos = np.array([rand_x, self.grid["B3"][1], 0])
            fast_dot.move_to(start_pos)
            
            stack_pos = get_stack_pos(rand_x)
            
            # Use a slight delay for each dot
            delay = i * 0.04
            # We add the dots to the scene but they are initially invisible or moved in Succession
            anim = Succession(
                Wait(delay),
                FadeIn(fast_dot, run_time=0.1),
                fast_dot.animate.move_to(stack_pos).set_run_time(0.2)
            )
            fast_dots_animations.append(anim)

        # Render the rapid sampling
        self.play(AnimationGroup(*fast_dots_animations))
        self.wait(3)
