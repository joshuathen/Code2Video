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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "The Intuition: Selective Focus"
        lines = [
            "Attention is like a flashlight in a dark room.",
            "It lets models focus on what truly matters.",
            "We prioritize relevant information over background noise."
        ]
        self.setup_layout(title, lines)

        # Time tracker for updaters
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Generate noisy dots field
        dots = VGroup()
        for _ in range(25):
            dot = Dot(color="#808080", radius=0.04)
            # Random position within the grid area B1 to F6
            rand_x = random.uniform(0.5, 5.5)
            rand_y = random.uniform(-2.8, 1.5)
            dot.move_to([rand_x, rand_y, 0])
            dots.add(dot)
            
        green_dot = Dot(color="#00FF00", radius=0.08)
        # Apply Issue 25: Centered flashlight
        self.place_in_area(green_dot, "C3", "C4", scale_factor=1.0)
        
        highlight_circle = Circle(radius=0.25, color="#00FF00", stroke_width=2)
        highlight_circle.add_updater(lambda m: m.move_to(green_dot.get_center()))
        
        # Jitter function
        def jitter_updater(mobj, dt):
            t = time_tracker.get_value()
            noise_x = np.sin(t * 3 + mobj.get_x() * 10) * 0.006
            noise_y = np.cos(t * 3 + mobj.get_y() * 10) * 0.006
            mobj.shift([noise_x, noise_y, 0])

        for d in dots:
            d.add_updater(jitter_updater)
        green_dot.add_updater(jitter_updater)
        
        self.play(FadeIn(dots), FadeIn(green_dot))
        self.play(Create(highlight_circle))
        self.wait(2)
        
        # === Animation for Lecture Line 2 ===
        # Update lecture colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#90EE90")
        )
        
        # Transition out first animation
        self.play(FadeOut(dots), FadeOut(green_dot), FadeOut(highlight_circle))
        
        # Frog and Fly animation
        frog = Circle(radius=0.4, color="#90EE90", fill_opacity=0.8)
        self.place_at_grid(frog, "C2")
        frog_label = Text("Frog", font_size=16, color="#90EE90").next_to(frog, DOWN, buff=0.2)
        
        fly = Dot(color="#FF6347", radius=0.1)
        self.place_at_grid(fly, "C5")
        fly_label = Text("Fly", font_size=16, color="#FF6347").next_to(fly, DOWN, buff=0.2)
        
        beam = Line(frog.get_right(), fly.get_left(), color="#FFFFFF", stroke_width=3)
        beam.set_stroke(opacity=0.6)
        
        self.play(FadeIn(frog), FadeIn(frog_label), FadeIn(fly), FadeIn(fly_label))
        self.play(Create(beam))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update lecture colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Transition out second animation
        self.play(FadeOut(frog), FadeOut(frog_label), FadeOut(fly), FadeOut(fly_label), FadeOut(beam))
        
        # Sequence of words boxes and central Meaning node
        boxes = VGroup()
        box_labels = ["The", "frog", "sees", "the", "fly"]
        grid_pos = ["E1", "E2", "E3", "E4", "E5"]
        
        for i in range(5):
            box_rect = Square(side_length=0.7, color=WHITE)
            self.place_at_grid(box_rect, grid_pos[i])
            box_text = Text(box_labels[i], font_size=14).move_to(box_rect.get_center())
            boxes.add(VGroup(box_rect, box_text))
            
        meaning_node = Circle(radius=0.5, color="#FFD700", fill_opacity=0.4)
        meaning_text = Text("Meaning", font_size=18, color="#FFD700").move_to(meaning_node.get_center())
        node_group = VGroup(meaning_node, meaning_text)
        
        # Apply Issue 24: Vertically adjusted meaning node
        self.place_in_area(node_group, 'C3', 'C4', scale_factor=1.0)
        
        lines_to_node = VGroup()
        for box in boxes:
            l = Line(box.get_top(), meaning_node.get_bottom(), color="#FFD700", stroke_width=1.5)
            l.set_stroke(opacity=0.5)
            lines_to_node.add(l)
            
        self.play(FadeIn(boxes))
        self.play(FadeIn(node_group))
        self.play(Create(lines_to_node))
        
        self.wait(3)
