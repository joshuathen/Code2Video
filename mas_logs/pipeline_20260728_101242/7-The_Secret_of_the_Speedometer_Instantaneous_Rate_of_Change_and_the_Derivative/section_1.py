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
        self.setup_layout(
            "The Cheetah’s Paradox: Average vs. Instant", 
            [
                "A cheetah sprints across the savanna.",
                "Average speed is total distance over total time.",
                "But how fast is it at one exact moment?",
                "A speedometer shows speed at a single frozen instant.",
                "This is the mystery of instantaneous rate of change."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Animate a stylized cheetah icon moving across the screen from 0 to 100 on a scale.
        self.lecture[0].set_color(WHITE)
        
        # Number line for the scale
        # Constructing scale manually to ensure reliability and avoid LaTeX issues
        scale_line = Line(self.grid["C1"], self.grid["C6"], color=WHITE)
        ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=WHITE).move_to(scale_line.point_from_proportion(i/5))
            for i in range(6)
        ])
        scale_labels = VGroup(*[
            Text(str(i*20), font_size=16).next_to(ticks[i], DOWN, buff=0.1)
            for i in range(6)
        ])
        scale = VGroup(scale_line, ticks, scale_labels)
        self.add(scale)
        
        # Cheetah icon (stylized as a triangle for speed)
        cheetah = Triangle(color=GOLD, fill_opacity=1).scale(0.15).rotate(-PI/2)
        start_pos = scale_line.get_start() + UP*0.3
        end_pos = scale_line.get_end() + UP*0.3
        cheetah.move_to(start_pos)
        
        self.play(FadeIn(cheetah))
        self.play(cheetah.animate.move_to(end_pos), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Display a horizontal line labeled '100 meters' (#FFFFFF) and a timer finishing at '4.0s' (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        
        # Reset cheetah for clear demonstration
        cheetah.move_to(start_pos)
        
        # Line for total distance
        dist_line = Line(scale_line.get_start(), scale_line.get_end(), color=WHITE).shift(DOWN*0.6)
        dist_label = Text("100 meters", font_size=18, color=WHITE)
        # Fix for Issue 21: Place in area D1-D6
        self.place_in_area(dist_label, "D1", "D6", scale_factor=1.0)
        dist_label.shift(DOWN*0.2)
        
        timer_val = ValueTracker(0)
        # Persistent DecimalNumber for the timer
        timer_text = DecimalNumber(0, num_decimal_places=1, color=WHITE)
        # Fix for Issue 22: Place at grid A5
        self.place_at_grid(timer_text, "A5", scale_factor=0.8)
        timer_text.add_updater(lambda m: m.set_value(timer_val.get_value()))
        timer_label = Text("Time:", font_size=18, color=WHITE).next_to(timer_text, LEFT, buff=0.2)
        
        self.add(timer_text, timer_label)
        
        self.play(
            FadeIn(dist_line),
            FadeIn(dist_label),
            timer_val.animate.set_value(4.0),
            cheetah.animate.move_to(end_pos),
            run_time=3,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Show the text 'Average Speed = 25 m/s' in #2ECC71 below the scale.
        self.lecture[2].set_color("#2ECC71")
        avg_speed_text = Text("Average Speed = 25 m/s", font_size=24, color="#2ECC71")
        self.place_in_area(avg_speed_text, "E1", "E6")
        
        self.play(Write(avg_speed_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Abruptly freeze the cheetah's motion at the 2.5s mark.
        self.lecture[3].set_color(WHITE)
        
        # Setup for freeze
        self.play(FadeOut(avg_speed_text), FadeOut(dist_line), FadeOut(dist_label))
        timer_val.set_value(0)
        cheetah.move_to(start_pos)
        
        # 2.5s at 25m/s = 62.5% of the distance (proportion 0.625)
        freeze_pos = scale_line.point_from_proportion(0.625) + UP*0.3
        
        self.play(
            timer_val.animate.set_value(2.5),
            cheetah.animate.move_to(freeze_pos),
            run_time=2.5,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the cheetah with a pulse effect in #FF0000 and label it 'Instant Speed?'.
        self.lecture[4].set_color("#FF0000")
        
        instant_label = Text("Instant Speed?", font_size=22, color="#FF0000")
        # Fix for Issue 23: Place in area B1-B6
        self.place_in_area(instant_label, "B1", "B6", scale_factor=1.0)
        
        pulse_circle = Circle(radius=0.1, color="#FF0000", stroke_width=4).move_to(cheetah.get_center())
        
        self.play(
            FadeIn(instant_label),
            pulse_circle.animate.scale(8).set_stroke(opacity=0),
            run_time=1.5
        )
        self.remove(pulse_circle)
        self.wait(2)
